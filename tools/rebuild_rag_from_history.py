"""mode/speaker 注釈済みの history.json を正本に memory.sqlite3 を作り直す。

手順（RAG 記憶の mode 正本化）の第3段。annotate_history_mode.py で付与し、人手で
2人だけモードを ``"mode":"two_only"``（＋assistant に ``"speaker"``）へ直した
history.json を読み、往復ペアを mode/speaker 付きでベクトル化して DB を再構築する。

旧 backfill は history.json の role/content しか見ず、全件 normal/main で投入して
いたため 1P/2P が混ざった。本ツールは history.json のエントリに付いた ``mode`` /
``speaker`` を読み、それを DB に正しく反映する。

スロット(slot)の決め方:
  - normal 行            → 常に 'main'（キャラ自身の1P会話）
  - two_only 行 かつ --main-name X 指定時
        speaker == X     → 'main'
        それ以外          → 'second'
  - two_only 行 で --main-name 未指定 → 'main'（従来同等の安全既定）
  ※ slot は recall のスロット絞り込みに使う。1P への 2P 混入防止は mode だけで
    達成されるため、slot 分割が不要なら --main-name は省略してよい。

使い方（app.py を動かす Python 環境で、リポジトリ直下から）:
  python tools/rebuild_rag_from_history.py --dry-run                 # 件数だけ確認
  python tools/rebuild_rag_from_history.py --reset                   # 全キャラ作り直し
  python tools/rebuild_rag_from_history.py --reset --char ruri       # 特定キャラのみ
  python tools/rebuild_rag_from_history.py --reset --char ruri --main-name ルリ
  python tools/rebuild_rag_from_history.py --test "君に作った料理"    # 投入後に想起テスト

注意:
  - ``--reset`` を付けないと既存 DB に追記されて重複するため、作り直しは必ず
    ``--reset`` を使うこと（付け忘れ防止に、--reset も --dry-run も無い場合は中断）。
  - 依存(numpy/torch or fastembed)が必要。rag_memory がフォールバック無効化されて
    いれば is_ready() が False を返し、埋め込みできないので中断する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import rag_memory as rag  # noqa: E402

SESSION_ROOT = _ROOT / "profiles" / "sessions"
VALID_MODES = {"normal", "two_only"}


def _read_entries(history_file: Path) -> list[dict]:
    """history.json から role/content/mode/speaker を持つエントリ列を取り出す。"""
    try:
        data = json.loads(history_file.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    history = data.get("history") if isinstance(data, dict) else data
    if not isinstance(history, list):
        return []
    entries: list[dict] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        mode = str(item.get("mode") or "").strip()
        entries.append(
            {
                "role": role,
                "content": content,
                "mode": mode if mode in VALID_MODES else "normal",
                "speaker": str(item.get("speaker") or "").strip(),
            }
        )
    return entries


def _iter_pairs(entries: list[dict]):
    """直近の user（お題）に対する各 assistant 返答を往復として返す。

    返すのは (user_text, reply_text, mode, speaker)。2人だけモードは「お題1つに
    ルリ→ユリカ→…と複数の返答が続く」ため、user は消費せず、後続の assistant も
    同じ直近 user に紐付ける（2つ目以降の話者を取りこぼさない）。通常の 1P は
    user/assistant が交互なので、この方式でも実質 1:1 のままになる。
    mode は「お題側・返答側のどちらかが two_only なら two_only」。speaker は返答側。
    """
    last_user: dict | None = None
    for entry in entries:
        if entry["role"] == "user":
            last_user = entry
        elif entry["role"] == "assistant" and last_user is not None:
            mode = (
                "two_only"
                if "two_only" in (last_user["mode"], entry["mode"])
                else "normal"
            )
            yield (last_user["content"], entry["content"], mode, entry["speaker"])


def _slot_for(mode: str, speaker: str, main_name: str) -> str:
    if mode != "two_only":
        return "main"
    if not main_name:
        return "main"  # 分割指定が無ければ従来同等
    return "main" if speaker == main_name else "second"


def _target_char_ids(only: list[str]) -> list[str]:
    if not SESSION_ROOT.exists():
        return []
    wanted = {str(c).strip() for c in only if str(c).strip()} or None
    ids: list[str] = []
    for char_dir in sorted(SESSION_ROOT.iterdir()):
        if not char_dir.is_dir() or not (char_dir / "history.json").exists():
            continue
        if wanted is not None and char_dir.name not in wanted:
            continue
        ids.append(char_dir.name)
    return ids


def _reset_db(char_id: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        target = Path(str(rag._db_path(char_id)) + suffix)
        target.unlink(missing_ok=True)


def rebuild(char_ids: list[str], *, reset: bool, dry_run: bool, main_name: str) -> None:
    grand = {"normal": 0, "two_only": 0, "fail": 0}
    for char_id in char_ids:
        entries = _read_entries(SESSION_ROOT / char_id / "history.json")
        pairs = list(_iter_pairs(entries))
        if reset and not dry_run:
            _reset_db(char_id)

        counts = {"normal": 0, "two_only": 0, "fail": 0}
        for user_text, reply_text, mode, speaker in pairs:
            slot = _slot_for(mode, speaker, main_name)
            if dry_run:
                counts[mode] += 1
                continue
            ok = rag.save_memory(char_id, slot, user_text, reply_text, mode=mode, speaker=speaker)
            counts[mode if ok else "fail"] += 1

        for key in grand:
            grand[key] += counts[key]
        total = "(dry-run)" if dry_run else _row_count(char_id)
        print(
            f"[{char_id}] pairs={len(pairs)} normal={counts['normal']} "
            f"two_only={counts['two_only']} fail={counts['fail']} db_total={total}"
        )
    print(
        f"\n== 再構築{'(dry-run)' if dry_run else '完了'}: "
        f"normal={grand['normal']} two_only={grand['two_only']} fail={grand['fail']} =="
    )


def _row_count(char_id: str) -> int:
    import sqlite3

    path = rag._db_path(char_id)
    if not path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(path))
        try:
            return int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def run_test(char_ids: list[str], query: str) -> None:
    print(f"\n== 想起テスト: query={query!r} ==")
    for char_id in char_ids:
        hits = rag.recall_memory(char_id, query)
        print(f"\n[{char_id}] hits={len(hits)}")
        block = rag.build_memory_block(hits)
        if block:
            print(block)


def main() -> int:
    parser = argparse.ArgumentParser(description="mode 注釈済み history.json から memory.sqlite3 を作り直す")
    parser.add_argument("--char", action="append", default=[], help="対象キャラID（複数可、未指定で全て）")
    parser.add_argument("--reset", action="store_true", help="対象キャラの memory.sqlite3 を消して作り直す")
    parser.add_argument("--dry-run", action="store_true", help="投入せず mode 別件数だけ表示")
    parser.add_argument("--main-name", default="", help="2Pで main スロット扱いにする話者名（例: ルリ）")
    parser.add_argument("--test", metavar="QUERY", default="", help="投入後に各キャラで想起テスト")
    args = parser.parse_args()

    if not rag.RAG_ENABLED:
        print("RAG が無効化されています（RAG_MEMORY_ENABLED=0）。", file=sys.stderr)
        return 2
    if not args.reset and not args.dry_run:
        print(
            "作り直しは --reset が必須です（付けないと既存 DB に重複追記されます）。\n"
            "件数だけ見たいなら --dry-run を使ってください。",
            file=sys.stderr,
        )
        return 2
    if not args.dry_run and not rag.is_ready():
        st = rag.status()
        print(
            "埋め込みモデルを初期化できませんでした。requirements-rag.txt を導入してください。\n"
            f"  詳細: {st.get('error') or 'unknown'}",
            file=sys.stderr,
        )
        return 2

    char_ids = _target_char_ids(args.char)
    if not char_ids:
        where = f"（--char {', '.join(args.char)}）" if args.char else ""
        print(f"対象の history.json が見つかりませんでした{where}: {SESSION_ROOT}", file=sys.stderr)
        return 1

    print(f"対象キャラ: {', '.join(char_ids)}")
    rebuild(char_ids, reset=args.reset, dry_run=args.dry_run, main_name=args.main_name.strip())
    if args.test and not args.dry_run:
        run_test(char_ids, args.test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
