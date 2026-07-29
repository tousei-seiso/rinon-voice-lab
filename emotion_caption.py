"""感情キャプション付き返答テキストの組み立て（app.py と tools/ の共通実装）。

``logs/chat.jsonl`` の各ターンは ``segments``（感情の区切りごとの
``{style, emoji, text}``）を持つ。表示用の注釈文「（😊嬉しそうに）本文……」は
この segments から機械的に組めるため、**chat.jsonl だけを正本にして復元できる**
（``logs/chat_emotion.jsonl`` の ``annotatedReply`` はこの関数の出力そのもので、
キャプションの独立した情報源ではない）。

実行時に記録する app.py と、chat.jsonl から作り直す tools/ が同じ整形を使うために
ここへ切り出した。片方だけ直すと再生成で注釈文が変わり、履歴が書き換わってしまう。
フロント表示 (static/app.js の buildAnnotatedReply) と同じ規則。
"""

from __future__ import annotations


def segments_have_text(segments: list[dict] | None) -> bool:
    """segments が分割本文(text)を持つか判定する。

    感情セグメント導入直後（2026-07-20）のごく一部のターンだけ ``style``/``emoji`` しか
    記録されておらず、そこからは注釈文を組めない。この形は
    ``tools/repair_chatlog_segments.py`` で ``chunks``/``audios`` から補修できる。
    """
    if not segments:
        return False
    return all(isinstance(seg.get("text"), str) for seg in segments)


def build_annotated_reply(reply: str, segments: list[dict] | None) -> str:
    """返答テキストの各セグメント先頭へ「（絵文字＋感情キャプション全文）」を挿入する。

    感情情報（分割本文またはキャプション）が無ければ元の ``reply`` をそのまま返す。
    """
    segments = segments or []
    has_marker = any(
        (str(seg.get("style") or "").strip() or str(seg.get("emoji") or "").strip())
        for seg in segments
    )
    if not segments_have_text(segments) or not has_marker:
        return reply
    parts: list[str] = []
    for seg in segments:
        marker = f"{str(seg.get('emoji') or '').strip()}{str(seg.get('style') or '').strip()}"
        text = str(seg.get("text") or "")
        parts.append(f"（{marker}）{text}" if marker else text)
    return "".join(parts)
