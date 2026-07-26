"""history.json の各エントリに ``mode`` を決め打ちで付与する準備用スクリプト。

手順（RAG 記憶の mode 正本化）の第1段:
  1) 本スクリプトで全エントリに ``"mode": "normal"`` を冪等に付与する。← ここ
  2) 人手で、2人だけモードの箇所だけ ``"mode": "two_only"`` に書き換え、
     さらに 2P の assistant 行へ ``"speaker": "ルリ"`` 等を書き足す。
  3) tools/rebuild_rag_from_history.py で history.json を正本に memory.sqlite3 を
     mode/speaker 付きで作り直す。

``mode`` は ``display`` と同じく LM 文脈では無視される表示外メタ。既に ``mode`` を
持つエントリは触らない（冪等）。編集事故に備え、初回のみ ``history.json.bak`` を残す。

使い方（app.py を動かす Python 環境で、リポジトリ直下から）:
  python tools/annotate_history_mode.py                 # 全キャラに付与
  python tools/annotate_history_mode.py --char ruri     # 特定キャラのみ（複数可）
  python tools/annotate_history_mode.py --dry-run       # 付与件数だけ確認（書き込まない）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Windows の cp932 コンソールでも日本語を出力できるよう標準出力を UTF-8 化する。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
SESSION_ROOT = _ROOT / "profiles" / "sessions"

VALID_MODES = {"normal", "two_only"}


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


def annotate_one(char_id: str, *, dry_run: bool) -> tuple[int, int]:
    """1 キャラの history.json に mode を付与する。(付与した数, 総エントリ数) を返す。"""
    history_file = SESSION_ROOT / char_id / "history.json"
    try:
        data = json.loads(history_file.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        print(f"[{char_id}] 読み込み失敗のためスキップ: {exc}", file=sys.stderr)
        return (0, 0)
    if not isinstance(data, dict) or not isinstance(data.get("history"), list):
        print(f"[{char_id}] history 配列が見つからないためスキップ", file=sys.stderr)
        return (0, 0)

    history = data["history"]
    added = 0
    for entry in history:
        if not isinstance(entry, dict):
            continue
        current = str(entry.get("mode") or "").strip()
        if current in VALID_MODES:
            continue  # 既に有効な mode を持つ → 触らない（冪等）
        entry["mode"] = "normal"
        added += 1

    if not dry_run and added:
        # 初回のみバックアップ（再実行で上書きしない）。
        bak = history_file.with_suffix(history_file.suffix + ".bak")
        if not bak.exists():
            shutil.copyfile(history_file, bak)
        history_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return (added, len(history))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="history.json の全エントリに mode='normal' を冪等付与する"
    )
    parser.add_argument("--char", action="append", default=[], help="対象キャラID（複数可、未指定で全て）")
    parser.add_argument("--dry-run", action="store_true", help="書き込まず件数だけ表示")
    args = parser.parse_args()

    char_ids = _target_char_ids(args.char)
    if not char_ids:
        where = f"（--char {', '.join(args.char)}）" if args.char else ""
        print(f"対象の history.json が見つかりませんでした{where}: {SESSION_ROOT}", file=sys.stderr)
        return 1

    print(f"対象キャラ: {', '.join(char_ids)}")
    grand_added = 0
    for char_id in char_ids:
        added, total = annotate_one(char_id, dry_run=args.dry_run)
        grand_added += added
        print(f"[{char_id}] entries={total} mode付与={added}")
    verb = "付与予定" if args.dry_run else "付与完了"
    print(f"\n== {verb}: {grand_added} 件（既に mode を持つ行は据え置き）==")
    if not args.dry_run:
        print("次: history.json を開き、2P会話の箇所を \"mode\": \"two_only\" に直し、")
        print("    2Pの assistant 行へ \"speaker\": \"ルリ\" 等を書き足してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
