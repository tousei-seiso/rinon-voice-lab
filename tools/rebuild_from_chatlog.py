"""logs/chat.jsonl を唯一の正本として、キャラ別 history.json と memory.sqlite3 を
まるごと再生成する。

背景: 品質に問題のあった過去の2人だけモード会話を chat.jsonl から手作業で除去した
うえで、その掃除済み chat.jsonl を「唯一の正本」に据え、他の履歴系ファイルを揃え直す
ための使い切りツール。chat.jsonl は各ターンに time/user/reply/speaker/twoOnlyMode
を持つため、元の時刻・会話モードを保ったまま再生成できる（history.json や旧DBには
時刻が無かった問題も同時に解消する）。

やること:
  1) chat.jsonl を1行=1ターンとして読み、speaker から char_id を決める
     （既定: ルリ→ruri, ユリカ→yurika。--map で上書き可）。
  2) char_id ごとに profiles/sessions/<charId>/history.json を再生成
     （role/content/mode/ts、返答には最低限の display）。感情キャプション付きの
     表示文は各ターンの ``segments`` から組み直す（下記「キャプションの復元」）。
  3) 同じ内容で memory.sqlite3 を作り直す（--reset、元時刻 ts 付き、mode 反映）。
  4) （任意）logs/chat_emotion.jsonl を chat.jsonl から作り直す（--sync-emotion）。

キャプションの復元:
  chat.jsonl の各ターンは ``segments``（``{style, emoji, text}`` の並び）を持つため、
  感情キャプション付きの表示文は **chat.jsonl だけから組み直せる**
  （``emotion_caption.build_annotated_reply``。chat_emotion.jsonl の ``annotatedReply``
  はこの関数の出力そのもので、独立した情報源ではない）。
  例外は感情セグメント導入当日（2026-07-20）のごく一部のターンで、segments に
  ``style``/``emoji`` しか無く text が欠けている。この形だけは chat_emotion.jsonl の
  対応表を予備の情報源として使い、それでも埋まらない場合は
  ``tools/repair_chatlog_segments.py`` で chunks/audios から text を補修できる。

使い方（app.py を動かす Python 環境で、リポジトリ直下から / .venv 有効）:
  python tools/rebuild_from_chatlog.py --dry-run
  python tools/rebuild_from_chatlog.py --reset
  python tools/rebuild_from_chatlog.py --reset --map "ルリ=ruri" --map "ユリカ=yurika"
  python tools/rebuild_from_chatlog.py --reset --sync-emotion --test "君に作った料理"

注意:
  - 破壊的操作なので、実行前に history.json は .regen.bak へ退避する（初回のみ）。
  - --reset も --dry-run も無い場合は誤操作防止のため中断する。
  - 未知の speaker があると（--map に無い）中断する（--skip-unmapped で読み飛ばし）。
  - 依存(numpy/torch or fastembed)が必要。埋め込み不可なら中断する。
  - **--reset は DB ファイルごと消すため、事実台帳(facts)も一緒に失われる。**
    再生成後に ``tools/build_fact_ledger.py`` を実行して台帳を作り直すこと
    （列挙質問の網羅と主客の取り違え防止に効く）。FTS5 語彙索引は自動で作られる。

会話を数件だけ削除／修正した場合は、全再生成ではなく
``tools/sync_memory.py``（差分同期）の方が速く、
台帳も保たれる。本ツールは chat.jsonl から全部作り直す場合に使う。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import rag_memory as rag  # noqa: E402
from emotion_caption import build_annotated_reply, segments_have_text  # noqa: E402

CHAT_LOG = _ROOT / "logs" / "chat.jsonl"
EMOTION_LOG = _ROOT / "logs" / "chat_emotion.jsonl"
SESSION_ROOT = _ROOT / "profiles" / "sessions"

# speaker 表示名 → キャラ別フォルダ(char_id) の既定対応。--map で上書き/追加できる。
DEFAULT_MAP = {"ルリ": "ruri", "ユリカ": "yurika"}


def _strip_speaker(text: str, names: set[str]) -> str:
    """返答文先頭の「名前: 」を外して素の返答テキストへ正規化する。"""
    base = str(text or "").strip()
    for name in sorted(names, key=len, reverse=True):
        m = re.match(rf"^{re.escape(name)}\s*[:：]\s*", base)
        if m:
            return base[m.end():].strip()
    return base


def _load_emotion_annotation(names: set[str]) -> dict[str, str]:
    """chat_emotion.jsonl から「素の返答 → 感情キャプション付き注釈文」の対応表を作る。

    **予備の情報源**。通常は chat.jsonl の segments から注釈文を組み直せる
    （build_annotated_reply）ので、この表は segments に分割本文(text)が欠けている
    ターン（感情セグメント導入当日のごく一部）にだけ使う。返答本文で引くため、
    同じ返答文が複数あると先着が優先される点にも注意（segments 経由なら起きない）。
    """
    annot: dict[str, str] = {}
    if not EMOTION_LOG.exists():
        return annot
    for line in EMOTION_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        raw = _strip_speaker(rec.get("reply"), names)
        annotated = str(rec.get("annotatedReply") or "")
        if raw and "（" in annotated and "）" in annotated and raw not in annot:
            annot[raw] = annotated
    return annot


def _read_chat_records() -> list[dict]:
    """chat.jsonl を1行=1ターンの dict 列として読む（壊れた行は飛ばす）。"""
    records: list[dict] = []
    bad = 0
    for line in CHAT_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            bad += 1
    if bad:
        print(f"  ※ 解析できなかった行を {bad} 件スキップしました。", file=sys.stderr)
    return records


def _parse_map(pairs: list[str]) -> dict[str, str]:
    mapping = dict(DEFAULT_MAP)
    for item in pairs:
        if "=" not in item:
            print(f"--map の書式が不正です（name=id 形式で）: {item!r}", file=sys.stderr)
            raise SystemExit(2)
        name, char_id = item.split("=", 1)
        name, char_id = name.strip(), char_id.strip()
        if name and char_id:
            mapping[name] = char_id
    return mapping


def _group_by_char(records: list[dict], mapping: dict[str, str], skip_unmapped: bool):
    """(char_id -> ターン列) と未知 speaker 集合を返す。ターンは chat.jsonl の並び順。"""
    grouped: dict[str, list[dict]] = {}
    unmapped: dict[str, int] = {}
    for rec in records:
        speaker = str(rec.get("speaker") or "").strip()
        user_text = str(rec.get("user") or "").strip()
        reply_text = str(rec.get("reply") or "").strip()
        if not user_text or not reply_text:
            continue  # 往復として成立しない行は除外
        char_id = mapping.get(speaker)
        if not char_id:
            unmapped[speaker] = unmapped.get(speaker, 0) + 1
            continue
        segments = rec.get("segments")
        grouped.setdefault(char_id, []).append(
            {
                "user": user_text,
                "reply": reply_text,
                "speaker": speaker,
                "ts": str(rec.get("time") or "").strip(),
                "mode": "two_only" if rec.get("twoOnlyMode") else "normal",
                "combinedUrl": str(rec.get("combinedUrl") or "").strip(),
                # 感情キャプションの復元に使う。chat.jsonl が正本たる所以なので
                # 必ず持ち越す（落とすと chat_emotion.jsonl 頼みに戻ってしまう）。
                "segments": segments if isinstance(segments, list) else [],
            }
        )
    return grouped, unmapped


def _annotated_for(turn: dict, annot: dict[str, str], names: set[str]) -> str:
    """1ターンの表示用テキスト（感情キャプション付き）を決める。

    優先順は chat.jsonl 自身 → chat_emotion.jsonl の対応表 → 素の返答。segments から
    組めるなら常にそちらを使う（ターン単位で厳密に対応し、返答本文の重複でも取り違えない）。
    """
    segments = turn.get("segments") or []
    if segments_have_text(segments):
        return build_annotated_reply(turn["reply"], segments)
    # segments に text が無い（導入当日のごく一部）ターンだけ予備の対応表へ落ちる。
    return annot.get(_strip_speaker(turn["reply"], names)) or turn["reply"]


def _write_history(char_id: str, turns: list[dict], annot: dict[str, str], names: set[str]) -> int:
    """char_id の history.json をターン列から再生成する。書いたエントリ数を返す。"""
    char_dir = SESSION_ROOT / char_id
    char_dir.mkdir(parents=True, exist_ok=True)
    history_file = char_dir / "history.json"
    if history_file.exists():
        bak = history_file.with_suffix(history_file.suffix + ".regen.bak")
        if not bak.exists():
            shutil.copyfile(history_file, bak)

    history: list[dict] = []
    for turn in turns:
        history.append(
            {"role": "user", "content": turn["user"], "mode": turn["mode"], "ts": turn["ts"]}
        )
        assistant = {
            "role": "assistant",
            "content": turn["reply"],
            "mode": turn["mode"],
            "ts": turn["ts"],
        }
        if turn["speaker"]:
            assistant["speaker"] = turn["speaker"]
        # 感情キャプション付きの表示文を display.text へ載せる。素の reply と変わらない
        # （＝キャプションが無い）ターンでは、音声 URL がある場合だけ display を作る。
        display_text = _annotated_for(turn, annot, names)
        if display_text != turn["reply"] or turn["combinedUrl"]:
            assistant["display"] = {
                "text": display_text,
                "meta": "",
                "audioUrl": turn["combinedUrl"],
            }
        history.append(assistant)

    payload = {
        "version": 2,
        "savedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "characterId": char_id,
        "history": history,
    }
    history_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(history)


def _reset_db(char_id: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(str(rag._db_path(char_id)) + suffix).unlink(missing_ok=True)


def _slot_for(mode: str, speaker: str, main_name: str) -> str:
    """保存先スロットを決める（sync_memory._slot_for と同じ規則）。

    --main-name 未指定なら常に 'main'（1P 前提の従来どおりの安全既定）。指定時は
    2人だけモードの往復だけ、その名前のキャラを 'main'、相手を 'second' に分ける。
    """
    if mode != "two_only" or not main_name:
        return "main"
    return "main" if speaker == main_name else "second"


def _rebuild_db(char_id: str, turns: list[dict], main_name: str = "") -> dict:
    counts = {"ok": 0, "fail": 0}
    for turn in turns:
        # mode/speaker/元時刻を保って保存する。slot は --main-name の指定に従う。
        ok = rag.save_memory(
            char_id,
            _slot_for(turn["mode"], turn["speaker"], main_name),
            turn["user"],
            turn["reply"],
            mode=turn["mode"],
            speaker=turn["speaker"],
            ts=turn["ts"],
        )
        counts["ok" if ok else "fail"] += 1
    # DB を作り直すと memories.id が振り直されるので、古い source_id を指したままの
    # 事実（別の往復・存在しない往復を指す）は掃除する。_reset_db を通っていれば
    # 台帳自体も消えているが、--reset 無しの経路でも整合するようここで必ず実行する。
    orphans = rag.prune_orphan_facts(char_id)
    if orphans:
        print(f"[{char_id}] 出典を失った事実を削除: {orphans} 件")
    return counts


def _pair_key(user_text: object, reply_text: object) -> tuple[str, str]:
    """時刻で照合できないとき用の予備キー（空白差を無視した ユーザー発言＋返答）。"""
    squash = lambda text: re.sub(r"\s+", "", str(text or ""))  # noqa: E731
    return (squash(user_text), squash(reply_text))


def _sync_emotion_log(records: list[dict], *, dry_run: bool = False) -> dict:
    """logs/chat_emotion.jsonl を chat.jsonl から作り直す。

    chat.jsonl が正本なので、消えた返答の行を落とすだけでなく **記録が欠けている
    ターンを足し、annotatedReply を segments から組み直す**。従来は「残る返答だけへ
    絞る」削る一方の処理だったため、欠けを補えず、このファイルを失うと復元できなかった。

    対象は app.py が記録するのと同じ「segments を持つターン」だけ。感情セグメント
    導入前のターンにはキャプションが存在しないので、無理に行を作らない。
    既存行の ``speakerSlot`` など chat.jsonl に無い項目は、照合できた行から引き継ぐ。
    """
    existing_by_time: dict[str, dict] = {}
    existing_by_pair: dict[tuple[str, str], dict] = {}
    before = 0
    if EMOTION_LOG.exists():
        for line in EMOTION_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if not s:
                continue
            before += 1
            try:
                rec = json.loads(s)
            except ValueError:
                continue
            stamp = str(rec.get("time") or "").strip()
            if stamp:
                existing_by_time.setdefault(stamp, rec)
            existing_by_pair.setdefault(_pair_key(rec.get("user"), rec.get("reply")), rec)

    out: list[dict] = []
    added = updated = unchanged = 0
    for rec in records:
        user_text = str(rec.get("user") or "").strip()
        reply_text = str(rec.get("reply") or "").strip()
        segments = rec.get("segments")
        if not user_text or not reply_text or not isinstance(segments, list) or not segments:
            continue
        stamp = str(rec.get("time") or "").strip()
        prior = existing_by_time.get(stamp) or existing_by_pair.get(
            _pair_key(user_text, reply_text)
        )
        entry = {
            "time": stamp,
            "user": user_text,
            "speaker": str(rec.get("speaker") or "").strip(),
        }
        slot = str((prior or {}).get("speakerSlot") or "").strip()
        if slot:
            entry["speakerSlot"] = slot  # chat.jsonl には無いので判る場合だけ引き継ぐ
        entry["model"] = str(rec.get("model") or (prior or {}).get("model") or "")
        entry["reply"] = reply_text
        entry["annotatedReply"] = build_annotated_reply(reply_text, segments)
        entry["segments"] = segments
        out.append(entry)
        if prior is None:
            added += 1
        elif prior.get("annotatedReply") != entry["annotatedReply"]:
            updated += 1
        else:
            unchanged += 1

    removed = max(0, before - (updated + unchanged))
    stats = {
        "before": before,
        "after": len(out),
        "added": added,
        "updated": updated,
        "removed": removed,
    }
    print(
        f"chat_emotion.jsonl: before={before} after={len(out)} "
        f"追加={added} 注釈更新={updated} 削除={removed}"
        + ("  (dry-run)" if dry_run else "")
    )
    if dry_run:
        return stats
    if EMOTION_LOG.exists():
        bak = EMOTION_LOG.with_suffix(EMOTION_LOG.suffix + ".regen.bak")
        if not bak.exists():
            shutil.copyfile(EMOTION_LOG, bak)
    EMOTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    EMOTION_LOG.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in out),
        encoding="utf-8",
    )
    return stats


def run_test(char_ids: list[str], query: str) -> None:
    print(f"\n== 想起テスト: query={query!r} ==")
    for char_id in char_ids:
        hits = rag.recall_memory(char_id, query, slot="main", mode="normal")
        print(f"\n[{char_id}] hits={len(hits)}")
        block = rag.build_memory_block(hits)
        if block:
            print(block)


def main() -> int:
    parser = argparse.ArgumentParser(description="chat.jsonl を正本に history.json と memory.sqlite3 を再生成")
    parser.add_argument("--map", action="append", default=[], help="speaker→char_id 対応（例: --map \"ルリ=ruri\"）")
    parser.add_argument("--reset", action="store_true", help="history.json と memory.sqlite3 を作り直す")
    parser.add_argument("--dry-run", action="store_true", help="書き込まず件数だけ表示")
    parser.add_argument("--skip-unmapped", action="store_true", help="未知 speaker を中断せず読み飛ばす")
    parser.add_argument(
        "--sync-emotion",
        "--filter-emotion",
        dest="sync_emotion",
        action="store_true",
        help="chat_emotion.jsonl も chat.jsonl から作り直す（--filter-emotion は旧名）",
    )
    parser.add_argument(
        "--main-name",
        default="",
        help="2人だけモードで main スロットに割り当てるキャラ名（未指定なら全て main）",
    )
    parser.add_argument("--test", metavar="QUERY", default="", help="再構築後に想起テスト")
    args = parser.parse_args()

    if not CHAT_LOG.exists():
        print(f"chat.jsonl が見つかりません: {CHAT_LOG}", file=sys.stderr)
        return 1
    if not args.reset and not args.dry_run:
        print("再生成は --reset が必須です（件数確認だけなら --dry-run）。", file=sys.stderr)
        return 2
    if not args.dry_run and not rag.is_ready():
        st = rag.status()
        print(f"埋め込みモデルを初期化できませんでした: {st.get('error') or 'unknown'}", file=sys.stderr)
        return 2

    mapping = _parse_map(args.map)
    records = _read_chat_records()
    grouped, unmapped = _group_by_char(records, mapping, args.skip_unmapped)

    print(f"chat.jsonl ターン数={sum(len(v) for v in grouped.values())}  対応={mapping}")
    if unmapped:
        print(f"未知 speaker: {unmapped}")
        if not args.skip_unmapped:
            print("→ --map で対応を指定するか --skip-unmapped を付けてください。中断します。", file=sys.stderr)
            return 2

    # 予備の対応表（segments に text が欠けているターンのキャプション復元にだけ使う）。
    names = set(mapping.keys())
    annot = _load_emotion_annotation(names)

    char_ids = sorted(grouped)
    for char_id in char_ids:
        turns = grouped[char_id]
        if args.dry_run:
            normal = sum(1 for t in turns if t["mode"] == "normal")
            two = sum(1 for t in turns if t["mode"] == "two_only")
            with_ts = sum(1 for t in turns if t["ts"])
            from_seg = sum(1 for t in turns if segments_have_text(t["segments"]))
            from_annot = sum(
                1
                for t in turns
                if not segments_have_text(t["segments"])
                and _strip_speaker(t["reply"], names) in annot
            )
            lost = sum(
                1
                for t in turns
                if t["segments"]
                and not segments_have_text(t["segments"])
                and _strip_speaker(t["reply"], names) not in annot
            )
            print(
                f"[{char_id}] turns={len(turns)} normal={normal} two_only={two} "
                f"ts有={with_ts} キャプション: segments={from_seg} 予備表={from_annot} 復元不能={lost}"
            )
            if lost:
                print(
                    "    ※ 復元不能分は tools/repair_chatlog_segments.py --dry-run で"
                    " 補修できるか確認できます。"
                )
            continue
        if args.reset:
            _reset_db(char_id)
        n_hist = _write_history(char_id, turns, annot, names)
        counts = _rebuild_db(char_id, turns, args.main_name)
        print(
            f"[{char_id}] turns={len(turns)} history={n_hist}行 "
            f"db_ok={counts['ok']} db_fail={counts['fail']}"
        )

    if args.sync_emotion:
        _sync_emotion_log(records, dry_run=args.dry_run)

    if args.test and not args.dry_run:
        run_test(char_ids, args.test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
