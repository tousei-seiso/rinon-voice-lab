"""時系列・列挙の想起が何を拾っているかを LLM 抜きで確認する読み取り専用の診断ツール。

「一番最初に買ってあげた本は何だっけ？」に妙な答えが返るとき、原因が
  ・意図検出（時系列質問だと気づいていない）
  ・ベクトル想起（最古の1件が上位枠から溢れている）
  ・語彙検索（語が一致していない／活用形の違い）
  ・事実台帳（未構築、または主客の向きを取り違えている）
  ・そもそも記録が無い（ts が潰れている、その往復が保存されていない）
のどれなのかを切り分ける。返答生成は行わないので LM Studio は不要。

使い方（app.py を動かす Python 環境 / .venv 有効、リポジトリ直下から）:
  python tools/diagnose_temporal.py --char ruri \
      --question "俺が君に一番最初に買ってあげた本は何だっけ？"

  # 実運用ではクエリを LLM が書き換えるので、その結果を手で与えて再現もできる
  python tools/diagnose_temporal.py --char ruri \
      --question "君に作ってあげた料理を全部挙げて" \
      --query "作った 料理 献立" --query "夕飯 おかず 手料理"

--show-block を付けると、実際に LLM へ渡るブロック（年表＋一覧）をそのまま表示する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import app  # noqa: E402
import fact_extract as fx  # noqa: E402
import rag_memory as rag  # noqa: E402


def diagnose(
    char_id: str,
    question: str,
    queries: list[str],
    *,
    slot: str,
    mode: str,
    user_name: str,
    char_name: str,
    show_block: bool,
) -> None:
    print(f"\n{'=' * 74}\n[{char_id}] 質問: {question}")

    span = rag.memory_span(char_id, slot=slot, mode=mode)
    print(
        f"  記録: {span['count']} 往復 / {rag.short_date(span['oldest']) or '?'} 〜 "
        f"{rag.short_date(span['newest']) or '?'}"
    )
    if span["count"] and not span["oldest"]:
        print("  警告: ts を解釈できる記録がありません（時系列想起は機能しません）。")
    stats = rag.facts_stats(char_id)
    print(
        f"  台帳: {stats['count']} 事実 / 抽出済み往復 {stats['sources']} / "
        f"方向別 {stats['directions']}"
    )
    print(
        f"        相別 {stats['modalities'] or '(相の列なし: 未再構築)'} / "
        f"出来事時刻あり {stats['occurred']} 件"
    )
    if stats["count"] and not stats["modalities"]:
        print(
            "  → 相（予定・否定）の列がまだありません。tools/build_fact_ledger.py "
            "--redo-rule で作り直すと、予定が「した事」に混ざらなくなります。"
        )
    if not stats["count"]:
        print("  → 台帳が空です。tools/build_fact_ledger.py で構築すると列挙・主客が安定します。")

    intent = app.detect_recall_intent(question)
    print(
        f"\n  意図検出: temporal={intent['temporal'] or '-'} enum={intent['enum']} "
        f"期間={intent['period'] or '-'} ({intent['since'] or '?'}〜{intent['until'] or '?'})"
    )
    filters = fx.infer_query_filters(question, user_name=user_name, char_name=char_name)
    print(f"  台帳の絞り込み: {filters}")
    if not filters["direction"]:
        print("  （向きが決められなかったので、両方向の事実を出します）")

    keywords: list[str] = []
    for query in queries:
        keywords.extend(rag.normalize_keywords(query))
    print(f"\n  語彙検索語（語幹化後）: {keywords or '(なし)'}")
    lexical = rag.search_lexical(
        char_id,
        keywords,
        slot=slot,
        mode=mode,
        since=intent["since"],
        until=intent["until"],
        order={"last": "newest"}.get(intent["temporal"], "oldest"),
        limit=12,
    )
    print(f"  語彙チャネル: {len(lexical)} 件（上位12件まで、古い順）")
    for hit in lexical:
        print(
            f"    {rag.format_stamp(hit['ts']) or '日時不明':17} hits={hit['hits']} "
            f"score={hit['score']:.3f} {app.compact_text(hit['user_text'], 44)}"
        )
    if not lexical and keywords:
        print("    → 一致なし。クエリの語が本文と違う（活用・言い換え）か、記録が無い可能性。")

    facts = rag.query_facts(
        char_id,
        category=filters["category"],
        verb=filters["verb"],
        direction=filters["direction"],
        modality=filters["modality"],
        slot=slot,
        mode=mode,
        since=intent["since"],
        until=intent["until"],
        order="newest" if intent["temporal"] == "last" else "oldest",
        limit=30,
    )
    print(f"\n  台帳チャネル: {len(facts)} 件"
        f"（相={filters['modality'] or 'すべて'} / 古い順、上位30件まで）")
    for fact in facts:
        when = f" 出来事={fact['occurred']}" if fact["occurred"] else ""
        if fact["time_hint"]:
            when += f"（{fact['time_hint']}）"
        print(
            f"    {rag.format_stamp(fact['ts'], seconds=True) or '(日時不明)':19} "
            f"{fact['direction']:11} {fact['modality']:8} "
            f"{fact['subject'] or '(不明)'}→{fact['recipient'] or '(不明)'} "
            f"{fact['verb']}: {fact['object']} [{fact['category'] or '-'}]{when} "
            f"conf={fact['confidence']:.2f} by={fact['extractor']}"
        )

    result = app.recall_for_turn(
        char_id,
        queries=queries,
        slot=slot,
        mode=mode,
        intent=intent,
        recent_user_texts=[],
        user_name=user_name,
        char_name=char_name,
    )
    print(f"\n  統合結果: {result['stats']} / 提示形式={result['mode'] or 'plain'}")
    if show_block:
        print(f"\n--- LLM へ渡るブロック ---\n{result['block'] or '(空)'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--char", action="append", default=[], help="対象キャラID（複数可）")
    parser.add_argument(
        "--question",
        default="俺が君に一番最初に買ってあげた本は何だっけ？",
        help="診断したい質問（そのままの発話）",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="想起に使う検索クエリ（省略時は質問文をそのまま使う）",
    )
    parser.add_argument("--slot", default="main", help="話者スロット（main/second）")
    parser.add_argument("--mode", default="normal", help="会話モード（normal/two_only）")
    parser.add_argument("--user-name", default="", help="ユーザーの呼び名（主客判定に使う）")
    parser.add_argument("--char-name", default="", help="キャラクター名（主客判定に使う）")
    parser.add_argument("--show-block", action="store_true", help="LLM へ渡るブロックを表示")
    args = parser.parse_args()

    session_root = _ROOT / "profiles" / "sessions"
    char_ids = [str(c).strip() for c in args.char if str(c).strip()]
    if not char_ids and session_root.exists():
        char_ids = [
            path.name
            for path in sorted(session_root.iterdir())
            if path.is_dir() and (path / "memory.sqlite3").exists()
        ]
    if not char_ids:
        print("対象キャラが見つかりません（profiles/sessions/<charId>/memory.sqlite3 が必要）。")
        return
    queries = [str(q).strip() for q in args.query if str(q).strip()] or [args.question]
    for char_id in char_ids:
        diagnose(
            char_id,
            args.question,
            queries,
            slot=args.slot,
            mode=args.mode,
            user_name=args.user_name,
            char_name=args.char_name,
            show_block=args.show_block,
        )


if __name__ == "__main__":
    main()
