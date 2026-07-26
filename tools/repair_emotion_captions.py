"""history.json のアシスタント返答から失われた感情キャプションを復元する。

背景:
  tools/rebuild_from_chatlog.py で history.json を chat.jsonl から再生成した際、
  display.text に「素の返答（reply）」をそのまま入れてしまい、各セグメント先頭に
  付いていた感情キャプション「（😊嬉しそうに…）」が丸ごと失われた。
  chat.jsonl の segments には style/emoji しか無く分割本文(text)が無いため、
  そこからは注釈文を再構成できない（これが取りこぼしの根本原因）。

  一方、注釈済みの完全な返答文は次のファイルに残っている:
    - logs/chat_emotion.jsonl        … reply(素) と annotatedReply(注釈付き) の対応
    - profiles/sessions/<char>/history.json.regen.bak … 再生成直前（注釈あり）
    - profiles/sessions/<char>/history.json.bak       … mode 付与直前（注釈あり）

  本ツールはこれらを「素の返答テキスト → 注釈付きテキスト」の対応表として使い、
  現行 history.json の各アシスタント返答の display.text に注釈を復元する。

やること:
  - 各アシスタント entry の content から話者接頭辞（例「ユリカ: 」）を外して素の
    返答テキストにし、対応表からキャプション付き注釈文を引く。
  - display.text に既に「（…）」キャプションがある entry は触らない（冪等）。
    キャプションが無い entry だけ、display.text を注釈付き文へ置き換える。
    display が無ければ {text, meta:"", audioUrl:""} を新規作成する。
  - content（LM 文脈）は素のまま一切変更しない。キャプションは表示用の display.text
    だけに載る（app.py の build_annotated_reply / sanitize_history と同じ設計）。

なぜ ruri は残って yurika は全滅に見えたか:
  同じ再生成で両者とも注釈を失っている。ruri はログが巨大で、そもそも感情セグメント
  を持たない旧発話や再生成後の実チャット（注釈あり）が多く混ざるため一部に注釈が
  残って見えた。yurika は注釈付き発話が全て（20件）再生成対象で、素の11件以外に
  注釈保持発話が無かったため 0 件に見えた、という違い。

使い方（app.py を動かす Python 環境で、リポジトリ直下から / .venv 有効）:
  python tools/repair_emotion_captions.py --dry-run            # 全キャラ、件数だけ
  python tools/repair_emotion_captions.py --char yurika        # yurika だけ復元
  python tools/repair_emotion_captions.py                      # 全キャラ復元

注意:
  - 書き込み前に history.json を history.json.caption-repair.bak へ退避（初回のみ）。
  - --dry-run なら一切書き込まず、復元可能件数だけを表示する。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# Windows の cp932 コンソールでも絵文字入り日本語を出力できるよう UTF-8 化する。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
SESSION_ROOT = _ROOT / "profiles" / "sessions"
EMOTION_LOG = _ROOT / "logs" / "chat_emotion.jsonl"

# 話者接頭辞「名前: 」を content から外すための既知話者名。emotion ログの speaker から
# も動的に補うが、最低限これらは固定で剥がす。
_KNOWN_SPEAKERS = {"ルリ", "ユリカ"}
_PREFIX_RE_CACHE: dict[frozenset, re.Pattern] = {}


def _prefix_re(names: frozenset) -> re.Pattern:
    pat = _PREFIX_RE_CACHE.get(names)
    if pat is None:
        alt = "|".join(sorted((re.escape(n) for n in names), key=len, reverse=True))
        pat = re.compile(rf"^(?:{alt})\s*[:：]\s*") if alt else re.compile(r"(?!)")
        _PREFIX_RE_CACHE[names] = pat
    return pat


def _strip_speaker(text: str, names: frozenset) -> str:
    """content 先頭の「名前: 」を外して素の返答テキストへ正規化する。"""
    return _prefix_re(names).sub("", str(text or "").strip()).strip()


def _has_caption(text: str) -> bool:
    """感情キャプション「（…）」を含むか（全角括弧で判定）。"""
    t = str(text or "")
    return "（" in t and "）" in t


def _load_history(path: Path) -> list[dict] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    hist = data.get("history") if isinstance(data, dict) else None
    return hist if isinstance(hist, list) else None


def _load_emotion_speakers() -> set[str]:
    names: set[str] = set(_KNOWN_SPEAKERS)
    if not EMOTION_LOG.exists():
        return names
    for line in EMOTION_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        sp = str(rec.get("speaker") or "").strip()
        if sp:
            names.add(sp)
    return names


def _build_lookup(char_id: str, names: frozenset) -> dict[str, str]:
    """素の返答テキスト → 注釈付きテキスト の対応表を作る。

    優先順位: chat_emotion.jsonl > history.json.regen.bak > history.json.bak。
    先に入った値を優先し、キャプションを含む値だけ採用する。
    """
    lookup: dict[str, str] = {}

    if EMOTION_LOG.exists():
        for line in EMOTION_LOG.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            raw = _strip_speaker(rec.get("reply"), names)
            annotated = rec.get("annotatedReply")
            if raw and _has_caption(annotated) and raw not in lookup:
                lookup[raw] = str(annotated)

    for tag in ("regen.bak", "bak"):
        path = SESSION_ROOT / char_id / f"history.json.{tag}"
        hist = _load_history(path)
        if not hist:
            continue
        for entry in hist:
            if not isinstance(entry, dict) or entry.get("role") != "assistant":
                continue
            display = entry.get("display")
            annotated = display.get("text") if isinstance(display, dict) else None
            raw = _strip_speaker(entry.get("content"), names)
            if raw and _has_caption(annotated) and raw not in lookup:
                lookup[raw] = str(annotated)

    return lookup


def repair_one(char_id: str, names: frozenset, *, dry_run: bool) -> tuple[int, int, int]:
    """1 キャラ分を復元する。(復元件数, 既に注釈あり, 対応表に無い) を返す。"""
    history_file = SESSION_ROOT / char_id / "history.json"
    data = None
    try:
        data = json.loads(history_file.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        print(f"[{char_id}] 読み込み失敗のためスキップ: {exc}", file=sys.stderr)
        return (0, 0, 0)
    history = data.get("history") if isinstance(data, dict) else None
    if not isinstance(history, list):
        print(f"[{char_id}] history 配列が無いためスキップ", file=sys.stderr)
        return (0, 0, 0)

    lookup = _build_lookup(char_id, names)
    restored = already = nomatch = 0

    for entry in history:
        if not isinstance(entry, dict) or entry.get("role") != "assistant":
            continue
        display = entry.get("display")
        current_text = display.get("text") if isinstance(display, dict) else None
        if _has_caption(current_text):
            already += 1
            continue  # 既に注釈あり → 触らない（冪等）
        raw = _strip_speaker(entry.get("content"), names)
        annotated = lookup.get(raw)
        if not annotated:
            nomatch += 1  # 対応表に無い = そもそも感情セグメントの無い発話
            continue
        if isinstance(display, dict):
            display["text"] = annotated
        else:
            entry["display"] = {"text": annotated, "meta": "", "audioUrl": ""}
        restored += 1

    if restored and not dry_run:
        bak = history_file.with_suffix(history_file.suffix + ".caption-repair.bak")
        if not bak.exists():
            shutil.copyfile(history_file, bak)
        history_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return (restored, already, nomatch)


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="history.json のアシスタント返答へ感情キャプションを復元する"
    )
    parser.add_argument("--char", action="append", default=[], help="対象キャラID（複数可、未指定で全て）")
    parser.add_argument("--dry-run", action="store_true", help="書き込まず復元可能件数だけ表示")
    args = parser.parse_args()

    char_ids = _target_char_ids(args.char)
    if not char_ids:
        where = f"（--char {', '.join(args.char)}）" if args.char else ""
        print(f"対象の history.json が見つかりません{where}: {SESSION_ROOT}", file=sys.stderr)
        return 1

    names = frozenset(_load_emotion_speakers())
    print(f"対象キャラ: {', '.join(char_ids)}  話者接頭辞: {', '.join(sorted(names))}")
    grand = 0
    for char_id in char_ids:
        restored, already, nomatch = repair_one(char_id, names, dry_run=args.dry_run)
        grand += restored
        verb = "復元可能" if args.dry_run else "復元"
        print(
            f"[{char_id}] {verb}={restored}  既に注釈あり={already}  "
            f"対応表に無し(=元々キャプション無し)={nomatch}"
        )
    tail = "（--dry-run のため書き込みなし）" if args.dry_run else ""
    print(f"\n== 合計 {grand} 件を{'復元予定' if args.dry_run else '復元'} {tail}==")
    if args.dry_run and grand:
        print("→ 実際に書き込むには --dry-run を外して再実行してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
