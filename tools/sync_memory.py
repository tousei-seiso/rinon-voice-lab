"""履歴を編集したあと、記憶系のファイルを**差分だけ**同期する統合ツール。

このプロジェクトの履歴系ファイルには上下関係がある。
    logs/chat.jsonl                              ← 一番大本の正本（全ターンの生ログ）
      ├→ profiles/sessions/<charId>/history.json   （画面復元用のキャラ別履歴）
      ├→ profiles/sessions/<charId>/memory.sqlite3 （RAG 検索DB＋事実台帳）
      └→ logs/chat_emotion.jsonl                   （感情キャプション付き返答）
``tools/rebuild_from_chatlog.py`` は chat.jsonl を正本に下流を**全部作り直す**ツールで、
本ツールはその差分版。全往復の埋め込みを計算し直さないので速く、事実台帳も保つ。

2 つのモードがある。編集した場所に応じて選ぶ。

``--source chatlog``（既定 / chat.jsonl を手修正したとき）
    chat.jsonl を正本として下流をすべて揃える。従来の rebuild_from_chatlog と同じ系統。
      1) history.json を再生成（感情キャプション付きの display も復元）
      2) memory.sqlite3 を差分同期（消えた往復を削除／無い往復を追加／時刻を更新）
      3) chat_emotion.jsonl を chat.jsonl に残る返答だけへ整合（--filter-emotion）
      4) 出典を失った事実を掃除し、追加分を台帳へ抽出（--extract）

``--source history``（history.json を手修正したとき）
    history.json を正本として memory.sqlite3 を差分同期する。
    **``--propagate`` を付けると、消えた往復を chat.jsonl と chat_emotion.jsonl からも
    削除する。** これを行わないと、一番大本の chat.jsonl には残ったままなので、
    後日 rebuild_from_chatlog を実行したときに削除した会話が復活する。

    なお **UI からの削除は、その時点で 4 系統すべて（history.json / chat.jsonl /
    chat_emotion.jsonl / memory.sqlite3＋台帳）へ反映される**ので、本ツールを流す必要は
    ない（app.py の /api/delete-turn が処理する）。このモードは history.json を直接
    編集した場合と、その反映に失敗した回を後から整合させる場合に使う。

使い方（app.py を動かす Python 環境 / .venv 有効、リポジトリ直下から）:
  # chat.jsonl を手修正した後（下流をすべて揃える）
  python tools/sync_memory.py --dry-run
  python tools/sync_memory.py --filter-emotion --extract --rule-only

  # UI で会話を削除した後（大本まで消して整合させる）
  python tools/sync_memory.py --source history --propagate --dry-run
  python tools/sync_memory.py --source history --propagate --extract --rule-only

注意:
  - --propagate は chat.jsonl を書き換えるため、実行前に .sync.bak へ退避する
    （rebuild_from_chatlog の .regen.bak、/api/delete-turn の .bak とは別名なので、
    互いに上書きしない）。
  - 往復の照合は「ユーザー発言＋返答＋会話モード」の一致で行う（ts は手修正されうる
    ので照合キーに含めない）。同じ往復が複数ある場合は出現回数で突き合わせる。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 同じ tools/ を読むため

import rag_memory as rag  # noqa: E402
import rebuild_from_chatlog as rfc  # noqa: E402  （chat.jsonl 読み込み・history 再生成を再利用）

SESSION_ROOT = _ROOT / "profiles" / "sessions"
CHAT_LOG = _ROOT / "logs" / "chat.jsonl"
EMOTION_LOG = _ROOT / "logs" / "chat_emotion.jsonl"
VALID_MODES = {"normal", "two_only"}


def _norm(text: object) -> str:
    """照合用の正規化（空白差だけの違いを同一視する）。"""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _key(user_text: object, reply_text: object, mode: object, names: set[str] | None = None) -> tuple:
    """往復の照合キー。返答は「話者名: 」を外した素の本文で比べる。

    保存場所によって返答の形が違う（chat.jsonl は本文のみ、history.json は
    「話者名: 本文」）だけでなく、DB の中でも両形式が混在している
    （実測: 過去の投入経路の違いで、素の本文の行と接頭辞付きの行が混ざっていた）。
    接頭辞を外して比べないと同じ往復を別物と判定し、全件を削除＋再追加してしまう
    （埋め込みの再計算が全件走り、memories.id が振り直されて台帳の紐付けも壊れる）。
    """
    reply = str(reply_text or "")
    if names:
        reply = rfc._strip_speaker(reply, names)
    return (_norm(user_text), _norm(reply), str(mode or "normal"))


def _backup_once(path: Path) -> None:
    """初回だけ .sync.bak へ退避する（書き換え前の安全網）。"""
    backup = path.with_suffix(path.suffix + ".sync.bak")
    if path.exists() and not backup.exists():
        shutil.copyfile(path, backup)
        print(f"  退避: {backup.name}")


# --- 正本の読み取り ------------------------------------------------------------


def _pairs_from_history(char_id: str) -> list[dict]:
    """history.json から往復列を作る（直近 user に各 assistant を紐付ける）。

    2人だけモードは「お題 1 つに複数キャラの返答が続く」ため user は消費しない
    （rebuild_rag_from_history と同じ規則）。
    """
    path = SESSION_ROOT / char_id / "history.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    history = data.get("history") if isinstance(data, dict) else data
    if not isinstance(history, list):
        return []
    pairs: list[dict] = []
    last_user: dict | None = None
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        mode = str(item.get("mode") or "").strip()
        entry = {
            "content": content,
            "mode": mode if mode in VALID_MODES else "normal",
            "speaker": str(item.get("speaker") or "").strip(),
            "ts": str(item.get("ts") or "").strip(),
        }
        if role == "user":
            last_user = entry
        elif last_user is not None:
            pairs.append(
                {
                    "user_text": last_user["content"],
                    "reply_text": entry["content"],
                    "mode": "two_only"
                    if "two_only" in (last_user["mode"], entry["mode"])
                    else "normal",
                    "speaker": entry["speaker"],
                    "ts": last_user["ts"] or entry["ts"],
                }
            )
    return pairs


def _pairs_from_chatlog(turns: list[dict], names: set[str]) -> list[dict]:
    """chat.jsonl のターン列を、DB 同期で使う往復形式へ正規化する。

    history.json 側の返答は「話者名: 本文」の形で保存されるため、DB もその形で
    記録されている。照合が食い違わないよう、ここで同じ接頭辞を付ける。
    """
    pairs: list[dict] = []
    for turn in turns:
        reply = str(turn.get("reply") or "").strip()
        speaker = str(turn.get("speaker") or "").strip()
        bare = rfc._strip_speaker(reply, names)
        pairs.append(
            {
                "user_text": str(turn.get("user") or "").strip(),
                "reply_text": f"{speaker}: {bare}" if speaker else bare,
                "mode": turn.get("mode") or "normal",
                "speaker": speaker,
                "ts": str(turn.get("ts") or "").strip(),
            }
        )
    return pairs


# --- memory.sqlite3 の差分同期 -------------------------------------------------


def _slot_for(mode: str, speaker: str, main_name: str) -> str:
    if mode != "two_only" or not main_name:
        return "main"
    return "main" if speaker == main_name else "second"


def _sync_db(
    char_id: str,
    pairs: list[dict],
    *,
    main_name: str,
    dry_run: bool,
    names: set[str] | None = None,
) -> dict:
    """正本の往復列に合わせて memories を差分同期する（追加・削除・時刻更新）。"""
    wanted: dict[tuple, list[dict]] = defaultdict(list)
    for pair in pairs:
        wanted[_key(pair["user_text"], pair["reply_text"], pair["mode"], names)].append(pair)
    stored: dict[tuple, list[dict]] = defaultdict(list)
    for turn in rag.list_turns(char_id):
        stored[_key(turn["user_text"], turn["reply_text"], turn["mode"], names)].append(turn)

    to_add: list[dict] = []
    to_delete: list[int] = []
    to_retime: list[tuple[int, str, str]] = []
    for key, items in wanted.items():
        existing = stored.get(key, [])
        for index, pair in enumerate(items):
            if index < len(existing):
                turn = existing[index]
                new_ts = str(pair.get("ts") or "").strip()
                # 正本に時刻があり DB と食い違う場合だけ更新する
                # （正本に時刻が無い往復の時刻を消してしまわないため）。
                if new_ts and rag.ts_sort_key(new_ts) != rag.ts_sort_key(turn["ts"]):
                    to_retime.append((turn["id"], turn["ts"], new_ts))
            else:
                to_add.append(pair)
        for turn in existing[len(items) :]:
            to_delete.append(turn["id"])
    for key, items in stored.items():
        if key not in wanted:
            to_delete.extend(turn["id"] for turn in items)

    result = {
        "db_total": sum(len(v) for v in stored.values()),
        "add": len(to_add),
        "delete": len(to_delete),
        "retime": len(to_retime),
        "fail": 0,
        "orphan": 0,
    }
    for pair in to_add[:5]:
        print(f"    + {rag.short_date(pair['ts']) or '----------'} {_norm(pair['user_text'])[:40]}")
    if len(to_add) > 5:
        print(f"    + …ほか {len(to_add) - 5} 件")
    for row_id in to_delete[:5]:
        print(f"    - id={row_id}")
    if len(to_delete) > 5:
        print(f"    - …ほか {len(to_delete) - 5} 件")
    for row_id, old_ts, new_ts in to_retime[:5]:
        print(f"    ~ id={row_id} {rag.short_date(old_ts) or '?'} → {rag.short_date(new_ts)}")
    if len(to_retime) > 5:
        print(f"    ~ …ほか {len(to_retime) - 5} 件")
    if dry_run:
        return result

    result["delete"] = rag.delete_memories(char_id, to_delete) if to_delete else 0
    retimed = 0
    for row_id, _old, new_ts in to_retime:
        if rag.update_turn_ts(char_id, row_id, new_ts):
            retimed += 1
    result["retime"] = retimed
    added = 0
    for pair in to_add:
        row_id = rag.save_turn(
            char_id,
            _slot_for(pair["mode"], pair["speaker"], main_name),
            pair["user_text"],
            pair["reply_text"],
            mode=pair["mode"],
            speaker=pair["speaker"],
            ts=pair["ts"],
        )
        if row_id:
            added += 1
        else:
            result["fail"] += 1
    result["add"] = added
    result["orphan"] = rag.prune_orphan_facts(char_id)
    return result


# --- 逆伝播（history の削除を chat.jsonl / chat_emotion.jsonl へ反映）----------


def _propagate_deletions(
    kept_keys: set[tuple], mapping: dict[str, str], names: set[str], *, dry_run: bool
) -> dict:
    """正本(history)に残っていない往復を chat.jsonl と chat_emotion.jsonl から削除する。

    UI の削除は history.json にしか効かないため、一番大本の chat.jsonl には残る。
    そのままだと後日 rebuild_from_chatlog を流したときに復活してしまうので、
    ここで大本まで消して整合させる。判定は往復キー（ユーザー発言＋返答＋モード）。
    """
    stats = {"chat_removed": 0, "chat_kept": 0, "emotion_removed": 0, "emotion_kept": 0}
    if not CHAT_LOG.exists():
        print("  chat.jsonl が無いため逆伝播をスキップします。")
        return stats
    kept_lines: list[str] = []
    removed_replies: set[str] = set()
    for line in CHAT_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except ValueError:
            kept_lines.append(line)  # 壊れた行は判断できないので温存する
            stats["chat_kept"] += 1
            continue
        user_text = str(record.get("user") or "").strip()
        reply_text = str(record.get("reply") or "").strip()
        speaker = str(record.get("speaker") or "").strip()
        if not user_text or not reply_text:
            kept_lines.append(line)
            stats["chat_kept"] += 1
            continue
        # 対応キャラが --map に無い行は対象外（判定材料が無いので温存）。
        if speaker not in mapping:
            kept_lines.append(line)
            stats["chat_kept"] += 1
            continue
        mode = "two_only" if record.get("twoOnlyMode") else "normal"
        if _key(user_text, reply_text, mode, names) in kept_keys:
            kept_lines.append(line)
            stats["chat_kept"] += 1
        else:
            stats["chat_removed"] += 1
            removed_replies.add(bare)
    print(
        f"  chat.jsonl: 残す={stats['chat_kept']} 削除={stats['chat_removed']}"
        + ("  (dry-run)" if dry_run else "")
    )
    if not dry_run and stats["chat_removed"]:
        _backup_once(CHAT_LOG)
        CHAT_LOG.write_text(
            "\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8"
        )

    # chat_emotion.jsonl は「返答本文」で紐づくので、削除された返答の行を落とす。
    # 同じ返答文が別の往復にも残っている場合は消さない（誤って消さないよう残存側を優先）。
    if EMOTION_LOG.exists() and removed_replies:
        surviving = {
            _norm(rfc._strip_speaker(key[1], names)) for key in kept_keys
        }
        kept_emotion: list[str] = []
        for line in EMOTION_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except ValueError:
                kept_emotion.append(line)
                stats["emotion_kept"] += 1
                continue
            bare = _norm(rfc._strip_speaker(record.get("reply"), names))
            if bare and bare not in surviving and bare in {_norm(r) for r in removed_replies}:
                stats["emotion_removed"] += 1
            else:
                kept_emotion.append(line)
                stats["emotion_kept"] += 1
        print(
            f"  chat_emotion.jsonl: 残す={stats['emotion_kept']} 削除={stats['emotion_removed']}"
            + ("  (dry-run)" if dry_run else "")
        )
        if not dry_run and stats["emotion_removed"]:
            _backup_once(EMOTION_LOG)
            EMOTION_LOG.write_text(
                "\n".join(kept_emotion) + ("\n" if kept_emotion else ""), encoding="utf-8"
            )
    return stats


# --- 台帳の追従 ---------------------------------------------------------------


def _extract_ledger(char_id: str, options: argparse.Namespace) -> None:
    pending = len(rag.iter_unextracted_turns(char_id))
    if not pending:
        print("  台帳の未抽出: 0 往復（追従済み）")
        return
    print(f"  台帳の未抽出: {pending} 往復 → 抽出します")
    import build_fact_ledger

    build_fact_ledger.build(
        [char_id],
        dry_run=False,
        reset=False,
        rule_only=options.rule_only,
        limit=0,
        user_name=options.user_name,
        char_name=options.char_name,
        model=options.model,
        generation_mode=options.generation_mode,
        show=False,
    )


# --- モード別の同期 -----------------------------------------------------------


def sync_from_chatlog(options: argparse.Namespace) -> None:
    """chat.jsonl を正本に、history.json / memory.sqlite3 / chat_emotion.jsonl を揃える。"""
    if not CHAT_LOG.exists():
        print(f"chat.jsonl が見つかりません: {CHAT_LOG}")
        return
    mapping = rfc._parse_map(options.map or [])
    records = rfc._read_chat_records()
    grouped, unmapped = rfc._group_by_char(records, mapping, options.skip_unmapped)
    if unmapped:
        detail = ", ".join(f"{name}×{count}" for name, count in unmapped.items())
        print(f"未対応の speaker: {detail}")
        if not options.skip_unmapped:
            print("→ --map name=id で対応を指定するか --skip-unmapped を付けてください。中断します。")
            return
    names = set(mapping.keys())
    annot = rfc._load_emotion_annotation(names)
    targets = [c for c in sorted(grouped) if not options.char or c in set(options.char)]
    if not targets:
        print("対象キャラが見つかりません。")
        return

    grand = defaultdict(int)
    for char_id in targets:
        turns = grouped[char_id]
        print(f"\n[{char_id}] chat.jsonl={len(turns)} 往復")
        # 1) history.json を chat.jsonl から作り直す（正本が上流なので再生成が正しい）
        if options.dry_run:
            print("  history.json: 再生成予定 (dry-run)")
        else:
            written = rfc._write_history(char_id, turns, annot, names)
            print(f"  history.json: {written} エントリを再生成")
        # 2) memory.sqlite3 を差分同期
        pairs = _pairs_from_chatlog(turns, names)
        result = _sync_db(
            char_id, pairs, main_name=options.main_name, dry_run=options.dry_run, names=names
        )
        print(
            f"  memory.sqlite3: DB={result['db_total']} → 追加={result['add']} "
            f"削除={result['delete']} 時刻更新={result['retime']} 失敗={result['fail']} "
            f"孤児事実削除={result['orphan']}"
        )
        for key in ("add", "delete", "retime", "fail", "orphan"):
            grand[key] += result[key]
        # 3) 台帳の追従
        if options.extract and not options.dry_run:
            _extract_ledger(char_id, options)

    # 4) chat_emotion.jsonl の整合（chat.jsonl に残る返答だけ残す）
    if options.filter_emotion:
        if options.dry_run:
            print("\nchat_emotion.jsonl: 整合予定 (dry-run)")
        else:
            valid = {turn["reply"] for turns in grouped.values() for turn in turns}
            rfc._filter_emotion(valid)
    _summary("chatlog", grand, options.dry_run)


def sync_from_history(options: argparse.Namespace) -> None:
    """history.json を正本に memory.sqlite3 を揃え、必要なら大本へ削除を伝播する。"""
    if not SESSION_ROOT.exists():
        print("profiles/sessions が見つかりません。")
        return
    wanted = set(options.char) if options.char else None
    targets = [
        path.name
        for path in sorted(SESSION_ROOT.iterdir())
        if path.is_dir()
        and (path / "history.json").exists()
        and (wanted is None or path.name in wanted)
    ]
    if not targets:
        print("対象キャラが見つかりません（history.json が必要）。")
        return

    grand = defaultdict(int)
    kept_keys: set[tuple] = set()
    # 返答の「話者名: 」を外して照合するため、話者名の一覧を先に用意する。
    names = set(rfc._parse_map(options.map or []).keys())
    for char_id in targets:
        pairs = _pairs_from_history(char_id)
        print(f"\n[{char_id}] history.json={len(pairs)} 往復")
        if not pairs:
            print("  空のためスキップします（履歴が読めない可能性）。")
            continue
        result = _sync_db(
            char_id, pairs, main_name=options.main_name, dry_run=options.dry_run, names=names
        )
        print(
            f"  memory.sqlite3: DB={result['db_total']} → 追加={result['add']} "
            f"削除={result['delete']} 時刻更新={result['retime']} 失敗={result['fail']} "
            f"孤児事実削除={result['orphan']}"
        )
        for key in ("add", "delete", "retime", "fail", "orphan"):
            grand[key] += result[key]
        for pair in pairs:
            kept_keys.add(_key(pair["user_text"], pair["reply_text"], pair["mode"], names))
        if options.extract and not options.dry_run:
            _extract_ledger(char_id, options)

    if options.propagate:
        print("\n--- 大本への逆伝播（chat.jsonl / chat_emotion.jsonl）---")
        if wanted is not None:
            print(
                "  ⚠ --char で対象を絞っています。逆伝播は chat.jsonl 全体を見るため、"
                "指定外キャラの往復まで『履歴に無い』と判定され削除されます。"
                "逆伝播するときは --char を付けずに全キャラで実行してください。"
            )
            return
        mapping = rfc._parse_map(options.map or [])
        _propagate_deletions(
            kept_keys, mapping, set(mapping.keys()), dry_run=options.dry_run
        )
    elif grand["delete"]:
        print(
            "\nヒント: 削除は memory.sqlite3 にだけ反映されました。"
            " 一番大本の chat.jsonl には残っているため、後で"
            " tools/rebuild_from_chatlog.py を実行すると復活します。"
            " 大本まで消すなら --propagate を付けて実行してください。"
        )
    _summary("history", grand, options.dry_run)


def _summary(source: str, grand: dict, dry_run: bool) -> None:
    print(
        f"\n== 同期{'(dry-run)' if dry_run else '完了'} [正本={source}]: "
        f"追加={grand['add']} 削除={grand['delete']} 時刻更新={grand['retime']} "
        f"孤児事実削除={grand['orphan']} 失敗={grand['fail']} =="
    )
    if grand["fail"]:
        print(
            "⚠ 追加に失敗した往復があります（埋め込みバックエンドが無効な可能性）。"
            " python -c \"import rag_memory; print(rag_memory.status())\" で確認してください。"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("chatlog", "history"),
        default="chatlog",
        help="正本をどちらにするか（既定 chatlog＝logs/chat.jsonl）",
    )
    parser.add_argument("--char", action="append", default=[], help="対象キャラID（複数可）")
    parser.add_argument("--dry-run", action="store_true", help="変更せず差分だけ表示")
    parser.add_argument(
        "--propagate",
        action="store_true",
        help="--source history のとき、削除を chat.jsonl / chat_emotion.jsonl へも反映",
    )
    parser.add_argument(
        "--filter-emotion",
        action="store_true",
        help="--source chatlog のとき、chat_emotion.jsonl も整合させる",
    )
    parser.add_argument(
        "--map", action="append", default=[], help="speaker表示名=charId（例 ルリ=ruri）"
    )
    parser.add_argument(
        "--skip-unmapped", action="store_true", help="--map に無い speaker の行を読み飛ばす"
    )
    parser.add_argument(
        "--main-name", default="", help="2人だけモードで main スロットに割り当てるキャラ名"
    )
    parser.add_argument("--extract", action="store_true", help="同期後に台帳の未抽出分を抽出")
    parser.add_argument("--rule-only", action="store_true", help="--extract 時に LLM を使わない")
    parser.add_argument("--user-name", default="", help="ユーザーの呼び名（主客判定に使う）")
    parser.add_argument("--char-name", default="", help="キャラクター名（主客判定に使う）")
    parser.add_argument("--model", default="", help="抽出に使う LM Studio モデル")
    parser.add_argument("--generation-mode", default="prefill", help="抽出時の生成モード")
    options = parser.parse_args()

    if not rag.RAG_ENABLED:
        print("RAG が無効化されています（RAG_MEMORY_ENABLED=0）。", file=sys.stderr)
        return 2
    if options.propagate and options.source != "history":
        print("--propagate は --source history と一緒に使ってください。", file=sys.stderr)
        return 2
    if options.source == "chatlog":
        sync_from_chatlog(options)
    else:
        sync_from_history(options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
