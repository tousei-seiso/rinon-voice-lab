"""既存の会話履歴を RAG のベクトルDBへ一括登録する使い切りスクリプト。

運用テストを「最初から DB 検索が効く」状態にするための初期投入用。
一次ソースは稼働中アプリと同じ profiles/sessions/<charId>/history.json で、
投入先も同じ profiles/sessions/<charId>/memory.sqlite3。フォルダ名(charId)は
フロントが送る character.id と一致するため、投入後そのまま recall が効く。

使い方（app.py を動かす Python 環境で、リポジトリ直下から）:
  python tools/backfill_rag_memory.py                 # 追記（重複はスキップ）
  python tools/backfill_rag_memory.py --reset         # 既存 memory.sqlite3 を消して作り直し
  python tools/backfill_rag_memory.py --char rinon    # 特定キャラのみ（複数可: --char rinon --char luvia）
  python tools/backfill_rag_memory.py --dry-run       # 登録せず件数だけ確認
  python tools/backfill_rag_memory.py --test "映画の話"  # 投入後に各キャラで想起テスト

注意:
  - history.json にはモード情報が無いため、投入分はすべて通常モード('normal')
    として登録される（想起時は「ユーザー:/返答:」表示）。2人だけモードの
    「お題:」表示は、投入後の実運用で新規に積まれる会話から反映される。
  - 依存(fastembed/numpy)が必要。未導入なら requirements-rag.txt を先に入れる。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# Windows の cp932 コンソールでも日本語/絵文字を出力できるよう標準出力を UTF-8 化する
# （保存データは UTF-8 で正しいが、print 時に UnicodeEncodeError で落ちるのを防ぐ）。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# リポジトリ直下の rag_memory.py を import できるようにする。
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import rag_memory as rag  # noqa: E402

SESSION_ROOT = _ROOT / "profiles" / "sessions"


def _read_history(history_file: Path) -> list[dict]:
    """history.json から会話エントリ列（role/content）を取り出す。"""
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
        if role in {"user", "assistant"} and content:
            entries.append({"role": role, "content": content})
    return entries


def _iter_pairs(entries: list[dict]):
    """user → assistant の 1 往復ペアを取り出す。

    直前の最も新しい user 発言に対する assistant 返答をペアにする
    （assistant は直近の user に応答している、という自然な仮定）。
    """
    last_user: str | None = None
    for entry in entries:
        if entry["role"] == "user":
            last_user = entry["content"]
        elif entry["role"] == "assistant" and last_user is not None:
            yield last_user, entry["content"]
            last_user = None


def _existing_pairs(char_id: str) -> set[tuple[str, str]]:
    """既に DB にある (user_text, reply_text) の集合を返す（重複投入の回避用）。"""
    path = rag._db_path(char_id)
    if not path.exists():
        return set()
    try:
        conn = sqlite3.connect(str(path))
        try:
            rows = conn.execute("SELECT user_text, reply_text FROM memories").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return set()
    return {(str(u or "").strip(), str(r or "").strip()) for u, r in rows}


def _row_count(char_id: str) -> int:
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


def _target_char_ids(only: list[str]) -> list[str]:
    if not SESSION_ROOT.exists():
        return []
    wanted = {rag._safe_id(c) for c in only} if only else None
    ids: list[str] = []
    for char_dir in sorted(SESSION_ROOT.iterdir()):
        if not char_dir.is_dir() or not (char_dir / "history.json").exists():
            continue
        if wanted is not None and char_dir.name not in wanted:
            continue
        ids.append(char_dir.name)
    return ids


def backfill(char_ids: list[str], *, reset: bool, dry_run: bool) -> None:
    grand_added = grand_skipped = 0
    for char_id in char_ids:
        history_file = SESSION_ROOT / char_id / "history.json"
        entries = _read_history(history_file)
        pairs = list(_iter_pairs(entries))
        if reset and not dry_run:
            # WAL/SHM ごと消して作り直す。
            for suffix in ("", "-wal", "-shm"):
                target = Path(str(rag._db_path(char_id)) + suffix)
                target.unlink(missing_ok=True)

        existing = set() if reset else _existing_pairs(char_id)
        added = skipped = 0
        for user_text, reply_text in pairs:
            key = (user_text.strip(), reply_text.strip())
            if key in existing:
                skipped += 1
                continue
            if dry_run:
                added += 1
                existing.add(key)
                continue
            # history.json はモード不明のため通常モードで投入する。
            if rag.save_memory(char_id, "main", user_text, reply_text, mode="normal"):
                added += 1
                existing.add(key)
            else:
                skipped += 1

        grand_added += added
        grand_skipped += skipped
        total = "(dry-run)" if dry_run else _row_count(char_id)
        print(
            f"[{char_id}] pairs={len(pairs)} added={added} skipped={skipped} "
            f"db_total={total}"
        )

    verb = "登録予定" if dry_run else "登録完了"
    print(f"\n== {verb}: added={grand_added} skipped={grand_skipped} ==")


def run_test(char_ids: list[str], query: str) -> None:
    print(f"\n== 想起テスト: query={query!r} ==")
    for char_id in char_ids:
        hits = rag.recall_memory(char_id, query)
        print(f"\n[{char_id}] hits={len(hits)}")
        block = rag.build_memory_block(hits)
        if block:
            print(block)


def main() -> int:
    parser = argparse.ArgumentParser(description="既存履歴を RAG DB へ一括登録する使い切りスクリプト")
    parser.add_argument("--char", action="append", default=[], help="対象キャラID（複数指定可、未指定なら全キャラ）")
    parser.add_argument("--reset", action="store_true", help="対象キャラの既存 memory.sqlite3 を消して作り直す")
    parser.add_argument("--dry-run", action="store_true", help="登録せず件数だけ表示")
    parser.add_argument("--test", metavar="QUERY", default="", help="投入後に各キャラで想起テストを実行")
    args = parser.parse_args()

    if not rag.RAG_ENABLED:
        print("RAG が無効化されています（RAG_MEMORY_ENABLED=0）。", file=sys.stderr)
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
    backfill(char_ids, reset=args.reset, dry_run=args.dry_run)

    if args.test and not args.dry_run:
        run_test(char_ids, args.test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
