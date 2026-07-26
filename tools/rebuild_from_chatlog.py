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
     （role/content/mode/ts、返答には最低限の display）。
  3) 同じ内容で memory.sqlite3 を作り直す（--reset、元時刻 ts 付き、mode 反映）。
  4) （任意）logs/chat_emotion.jsonl を、掃除済み chat.jsonl に残る返答だけへ整合。

使い方（app.py を動かす Python 環境で、リポジトリ直下から / .venv 有効）:
  python tools/rebuild_from_chatlog.py --dry-run
  python tools/rebuild_from_chatlog.py --reset
  python tools/rebuild_from_chatlog.py --reset --map "ルリ=ruri" --map "ユリカ=yurika"
  python tools/rebuild_from_chatlog.py --reset --filter-emotion --test "君に作った料理"

注意:
  - 破壊的操作なので、実行前に history.json は .regen.bak へ退避する（初回のみ）。
  - --reset も --dry-run も無い場合は誤操作防止のため中断する。
  - 未知の speaker があると（--map に無い）中断する（--skip-unmapped で読み飛ばし）。
  - 依存(numpy/torch or fastembed)が必要。埋め込み不可なら中断する。
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

    chat.jsonl の segments には分割本文(text)が無く注釈文を再構成できないため、
    display.text へ載せるキャプション付き文はこの対応表から引く（これが無いと
    再生成で感情キャプションが失われる。tools/repair_emotion_captions.py 参照）。
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
        grouped.setdefault(char_id, []).append(
            {
                "user": user_text,
                "reply": reply_text,
                "speaker": speaker,
                "ts": str(rec.get("time") or "").strip(),
                "mode": "two_only" if rec.get("twoOnlyMode") else "normal",
                "combinedUrl": str(rec.get("combinedUrl") or "").strip(),
            }
        )
    return grouped, unmapped


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
        # 感情キャプション付きの表示文（あれば）を display.text へ載せる。素の reply では
        # なく chat_emotion.jsonl 由来の注釈文を使うことで、再生成でキャプションを失わない。
        annotated = annot.get(_strip_speaker(turn["reply"], names))
        display_text = annotated if annotated else turn["reply"]
        if annotated or turn["combinedUrl"]:
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


def _rebuild_db(char_id: str, turns: list[dict]) -> dict:
    counts = {"ok": 0, "fail": 0}
    for turn in turns:
        # 1P 前提のため slot は 'main'。mode/speaker/元時刻を保って保存する。
        ok = rag.save_memory(
            char_id,
            "main",
            turn["user"],
            turn["reply"],
            mode=turn["mode"],
            speaker=turn["speaker"],
            ts=turn["ts"],
        )
        counts["ok" if ok else "fail"] += 1
    return counts


def _filter_emotion(valid_replies: set[str]) -> None:
    """chat_emotion.jsonl を、掃除済み chat.jsonl に残る返答だけへ整合させる。"""
    if not EMOTION_LOG.exists():
        print("chat_emotion.jsonl が無いためスキップ")
        return
    lines = EMOTION_LOG.read_text(encoding="utf-8").splitlines()
    kept, removed = [], 0
    for line in lines:
        s = line.strip()
        if not s:
            continue
        try:
            rec = json.loads(s)
        except Exception:
            kept.append(line)  # 壊れた行は温存
            continue
        if str(rec.get("reply") or "").strip() in valid_replies:
            kept.append(line)
        else:
            removed += 1
    bak = EMOTION_LOG.with_suffix(EMOTION_LOG.suffix + ".regen.bak")
    if not bak.exists():
        shutil.copyfile(EMOTION_LOG, bak)
    EMOTION_LOG.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    print(f"chat_emotion.jsonl: before={len(lines)} removed={removed} after={len(kept)}")


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
    parser.add_argument("--filter-emotion", action="store_true", help="chat_emotion.jsonl も整合させる")
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

    # 素の返答 → 感情キャプション付き注釈文。display.text 復元に使う（話者名で接頭辞を剥がす）。
    annot = _load_emotion_annotation(set(mapping.keys()))

    char_ids = sorted(grouped)
    for char_id in char_ids:
        turns = grouped[char_id]
        if args.dry_run:
            normal = sum(1 for t in turns if t["mode"] == "normal")
            two = sum(1 for t in turns if t["mode"] == "two_only")
            with_ts = sum(1 for t in turns if t["ts"])
            captioned = sum(1 for t in turns if _strip_speaker(t["reply"], set(mapping.keys())) in annot)
            print(
                f"[{char_id}] turns={len(turns)} normal={normal} two_only={two} "
                f"ts有={with_ts} キャプション復元可={captioned}"
            )
            continue
        if args.reset:
            _reset_db(char_id)
        n_hist = _write_history(char_id, turns, annot, set(mapping.keys()))
        counts = _rebuild_db(char_id, turns)
        print(
            f"[{char_id}] turns={len(turns)} history={n_hist}行 "
            f"db_ok={counts['ok']} db_fail={counts['fail']}"
        )

    if not args.dry_run and args.filter_emotion:
        valid = {t["reply"] for turns in grouped.values() for t in turns}
        _filter_emotion(valid)

    if args.test and not args.dry_run:
        run_test(char_ids, args.test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
