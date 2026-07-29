"""logs/chat.jsonl の segments に欠けている分割本文(text)を chunks/audios から復元する。

背景:
  感情セグメント機能を入れた当日（2026-07-20）のごく一部のターンだけ、segments に
  ``style``/``emoji`` しか記録されていない（``text`` が無い）。この形だと
  ``emotion_caption.build_annotated_reply`` が注釈文を組めないため、chat.jsonl を
  正本にした再生成で感情キャプションを復元できない。しかも該当ターンは
  ``logs/chat_emotion.jsonl`` への記録が始まる前なので、そちらにも控えが無い。

  ただし chat.jsonl は同じターンに ``chunks``（実際に読み上げた文の並び）と
  ``audios[].emojiStyle``（チャンクごとの発声効果絵文字）を持っている。絵文字は
  セグメント単位で付くので、**絵文字が変わるところがセグメントの境界**になる。
  ここから分割本文を機械的に復元できる。

やること:
  1) segments があるのに text が無いターンを洗い出す。
  2) chunks を audios[].emojiStyle の変化で束ね、各セグメントの本文を組み立てる。
  3) 下の 4 点すべてを満たしたターンだけ書き込む（1つでも崩れたら触らない）:
       ・chunks と audios の件数が一致する
       ・束ねたグループ数が segments の件数と一致する
       ・グループの絵文字の並びが segments の絵文字の並びと一致する
       ・全チャンクを連結すると reply に戻る（空白差は無視）
     復元は推定なので、裏付けが取れない限り黙って埋めない方針。
  4) 他の行は 1 バイトも触らず、該当行だけ差し替えて書き戻す。

使い方（リポジトリ直下から。埋め込みも LLM も使わないので .venv でなくても動く）:
  python tools/repair_chatlog_segments.py --dry-run          # 復元できるか確認
  python tools/repair_chatlog_segments.py --dry-run --show    # 復元結果を全文表示
  python tools/repair_chatlog_segments.py --apply             # 書き込む

注意:
  - 書き込み前に ``logs/chat.jsonl.segrepair.bak`` へ退避する（初回のみ。他ツールの
    .bak / .regen.bak / .sync.bak とは別名なので互いに上書きしない）。
  - --apply も --dry-run も無い場合は誤操作防止のため中断する。
  - 補修後は下流へ反映するため
    ``python tools/sync_memory.py --sync-emotion --extract --rule-only``
    （または ``tools/rebuild_from_chatlog.py --reset --sync-emotion``）を流すこと。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from emotion_caption import build_annotated_reply, segments_have_text  # noqa: E402

CHAT_LOG = _ROOT / "logs" / "chat.jsonl"


def _squash(text: object) -> str:
    """空白を落として比べるための正規化（読み上げ用の整形差を無視する）。"""
    return re.sub(r"\s+", "", str(text or ""))


def _group_chunks_by_emoji(chunks: list[str], audios: list[dict]) -> list[tuple[str, list[str]]]:
    """チャンクを「発声効果絵文字が変わるところ」で束ねる。

    絵文字は app.py の render_reply_audio がセグメント単位で付ける。持続系の絵文字は
    セグメント内の全チャンクに付き、単発音は先頭チャンクだけに付いて残りは空になる
    （emoji_is_sustained）。そのため **空の絵文字は直前グループの続き**として扱う。
    """
    groups: list[tuple[str, list[str]]] = []
    for chunk, audio in zip(chunks, audios):
        emoji = str(audio.get("emojiStyle") or "")
        if groups and (emoji == groups[-1][0] or not emoji):
            groups[-1][1].append(chunk)
        else:
            groups.append((emoji, [chunk]))
    return groups


def _restore_segment_texts(record: dict) -> tuple[list[str] | None, str]:
    """1ターン分の分割本文を復元する。(本文の並び, 却下理由) を返す。"""
    segments = record.get("segments") or []
    chunks = [str(c) for c in (record.get("chunks") or [])]
    audios = record.get("audios") or []
    reply = str(record.get("reply") or "")

    if not chunks or not audios:
        return None, "chunks/audios が無い"
    if len(chunks) != len(audios):
        return None, f"chunks={len(chunks)} と audios={len(audios)} の件数が違う"
    groups = _group_chunks_by_emoji(chunks, audios)
    if len(groups) != len(segments):
        return None, f"グループ数={len(groups)} が segments={len(segments)} と違う"
    seg_emojis = [str(seg.get("emoji") or "") for seg in segments]
    if [emoji for emoji, _ in groups] != seg_emojis:
        return None, "グループの絵文字の並びが segments と一致しない"
    texts = ["".join(parts) for _, parts in groups]
    if _squash("".join(texts)) != _squash(reply):
        return None, "連結しても reply に戻らない"
    return texts, ""


def _iter_lines() -> list[str]:
    return CHAT_LOG.read_text(encoding="utf-8", errors="replace").splitlines()


def repair(*, dry_run: bool, show: bool) -> int:
    lines = _iter_lines()
    out_lines: list[str] = []
    repaired = rejected = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except ValueError:
            out_lines.append(line)  # 壊れた行は判断できないので温存する
            continue
        segments = record.get("segments") or []
        # 分割本文が揃っている行と、そもそも segments が無い行（機能導入前）は対象外。
        if not segments or segments_have_text(segments):
            out_lines.append(line)
            continue

        texts, reason = _restore_segment_texts(record)
        stamp = str(record.get("time") or "(日時不明)")
        if texts is None:
            rejected += 1
            print(f"  × {stamp} 復元できません: {reason}")
            out_lines.append(line)
            continue

        for seg, text in zip(segments, texts):
            seg["text"] = text
        repaired += 1
        print(f"  ○ {stamp} segments={len(segments)} を復元")
        if show:
            print(f"      {build_annotated_reply(str(record.get('reply') or ''), segments)}")
        out_lines.append(json.dumps(record, ensure_ascii=False))

    print(f"\n対象={repaired + rejected} 復元={repaired} 却下={rejected}")
    if not repaired:
        print("書き込む変更はありません。")
        return 0
    if dry_run:
        print("(dry-run のため書き込みません)")
        return 0

    backup = CHAT_LOG.with_suffix(CHAT_LOG.suffix + ".segrepair.bak")
    if not backup.exists():
        shutil.copyfile(CHAT_LOG, backup)
        print(f"退避: {backup.name}")
    CHAT_LOG.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"logs/chat.jsonl を書き換えました（{repaired} ターンを補修）。")
    print(
        "続けて下流へ反映してください: "
        "python tools/sync_memory.py --sync-emotion --extract --rule-only"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="chat.jsonl の segments に欠けている分割本文を chunks/audios から復元する"
    )
    parser.add_argument("--apply", action="store_true", help="実際に書き込む")
    parser.add_argument("--dry-run", action="store_true", help="書き込まず復元可否だけ表示")
    parser.add_argument("--show", action="store_true", help="復元した注釈文を全文表示")
    args = parser.parse_args()

    if not CHAT_LOG.exists():
        print(f"chat.jsonl が見つかりません: {CHAT_LOG}", file=sys.stderr)
        return 1
    if not args.apply and not args.dry_run:
        print("書き込みは --apply が必須です（確認だけなら --dry-run）。", file=sys.stderr)
        return 2
    return repair(dry_run=not args.apply, show=args.show)


if __name__ == "__main__":
    raise SystemExit(main())
