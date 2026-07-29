"""相（した事／予定／しなかった事）と出来事時刻の判定を例文で検証する回帰チェック。

台帳の誤記録で一番厄介なのは、**未来の予定や過去の回想が「そのときにした事」として
記録される**ことだった（実測: 「次はうどんなカルボナーラを作ってあげるね」が
*作った料理*として列挙され、「1年前に多摩川の花火大会に行ったよね」が*その話をした日*の
出来事になった）。この手の誤りは正規表現を触るたびに再発しうるので、代表例を固定して
毎回同じ判定になるかを確かめる。

LLM も DB も使わない（ルール抽出だけを見る）ので数百ミリ秒で終わる。
``fact_extract.py`` の語尾・時間表現の規則を触ったら実行すること。

使い方（リポジトリ直下から）:
  python tools/check_fact_modality.py           # 失敗した例だけを表示
  python tools/check_fact_modality.py --all     # 全例の判定を表示
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

import fact_extract as fx  # noqa: E402

# 抽出の基準日（相対表現の解決に使う）。固定値にしないと期待値が日々変わる。
TS = "2026-07-30 21:00:00"
USER = "オサム"
CHAR = "ルリ"

# (発話, 期待する (動詞, 客体, 相, 出来事時刻)) の列。
# 出来事時刻は前方一致の文字列（'' は「話した日と同じ扱い」）。
CASES: tuple[tuple[str, tuple[str, str, str, str]], ...] = (
    # --- 未来の予定を「した事」にしない（今回の主症状）---
    ("次はうどんなカルボナーラを作ってあげるね", ("作る", "うどんなカルボナーラ", "plan", "")),
    ("明日は肉じゃがを作ってあげる", ("作る", "肉じゃが", "plan", "")),
    ("今度カルボナーラを作ってあげるって約束したよね", ("作る", "カルボナーラ", "plan", "")),
    ("そのうち映画館に行こうと思ってる", ("行く", "映画館", "plan", "")),
    ("今度カルボナーラを作ろうね", ("作る", "カルボナーラ", "plan", "")),
    # 「〜しようとした」は試みであって完了ではない。plan かどうかより
    # 「done ではない」ことが大事（列挙に「作った料理」として出ないこと）。
    ("カルボナーラを作ろうとしたけど失敗した", ("作る", "カルボナーラ", "plan", "")),
    ("今度うどんなカルボナーラを作れたら作ってあげるよ", ("作る", "うどんなカルボナーラ", "wish", "")),
    ("オムライスを作ってあげたいな", ("作る", "オムライス", "wish", "")),
    ("カルボナーラを作れなかったんだ", ("作る", "カルボナーラ", "negated", "")),
    # --- 過去の回想は「した事」だが、時期は話した日ではない ---
    ("1年前に多摩川の花火大会に一緒に行ったよね", ("行く", "多摩川の花火大会", "done", "2025")),
    ("去年の夏に江ノ島へ行ったよね", ("行く", "江ノ島", "done", "2025-07")),
    ("3ヶ月前に鎌倉のカフェに行ったのが楽しかった", ("行く", "鎌倉のカフェ", "done", "2026-04")),
    ("2024年3月に水族館に行ったね", ("行く", "水族館", "done", "2024-03")),
    ("昨日カレーを作ってあげたよ", ("作る", "カレー", "done", "2026-07-29")),
    # --- 普通の過去（時期は話した日と同じなので occurred は空）---
    ("肉じゃがを作ってあげた", ("作る", "肉じゃが", "done", "")),
    ("ケーキを作って渡してくれたね", ("作る", "ケーキ", "done", "")),
    ("星の王子さまを買ってくれてありがとう", ("買う", "星の王子さま", "done", "")),
    ("いつも味噌汁を作ってくれてるね", ("作る", "味噌汁", "done", "")),
    ("肉じゃがを作って持って行ったよ", ("作る", "肉じゃが", "done", "")),
    # 動詞パターン自体が「た」を含む形（語尾に完了の印が残らない）。
    ("肉じゃがを煮たよ", ("作る", "肉じゃが", "done", "")),
    ("温泉宿に泊めたね", ("泊まる", "温泉宿", "done", "")),
)

# 質問文から推定する相（既定は done。予定を訊いていると読めるときだけ plan）。
QUESTION_CASES: tuple[tuple[str, str], ...] = (
    ("俺が作ってあげた料理を挙げて", "done"),
    ("君が作ってくれた料理を全部挙げて", "done"),
    ("次に作ったのは何だっけ", "done"),
    ("今度作ってくれるって言ってた料理は？", "plan"),
    ("まだ作ってない料理ある？", "plan"),
)


def _describe(fact: dict) -> str:
    when = fact.get("occurred") or "-"
    hint = fact.get("time_hint")
    return (
        f"{fact.get('verb')}:{fact.get('object')} "
        f"[{fact.get('direction')}/{fact.get('modality')}] 出来事={when}"
        + (f"（{hint}）" if hint else "")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="成功した例も表示する")
    args = parser.parse_args()
    failures = 0
    for text, expected in CASES:
        verb, obj, modality, occurred = expected
        facts = fx.extract_rule_based(
            text, "", user_name=USER, char_name=CHAR, ts=TS
        )
        hit = next(
            (
                fact
                for fact in facts
                if fact["verb"] == verb and fact["object"] == obj
            ),
            None,
        )
        if hit is None:
            failures += 1
            print(f"✗ {text}")
            print(f"    期待: {verb}:{obj}[{modality}] / 実際: 該当なし")
            for fact in facts:
                print(f"      （抽出されたのは） {_describe(fact)}")
            continue
        problems = []
        if hit["modality"] != modality:
            problems.append(f"相 {hit['modality']} ≠ {modality}")
        if hit["occurred"] != occurred:
            problems.append(f"出来事時刻 {hit['occurred'] or '(空)'} ≠ {occurred or '(空)'}")
        if problems:
            failures += 1
            print(f"✗ {text}")
            print(f"    {' / '.join(problems)} — {_describe(hit)}")
        elif args.all:
            print(f"✓ {text}\n    {_describe(hit)}")
    for question, expected in QUESTION_CASES:
        actual = fx.infer_query_filters(question, user_name=USER, char_name=CHAR)
        if actual["modality"] != expected:
            failures += 1
            print(f"✗ 質問「{question}」→ 相={actual['modality']} ≠ {expected}")
        elif args.all:
            print(f"✓ 質問「{question}」→ 相={actual['modality']}")
    total = len(CASES) + len(QUESTION_CASES)
    print(f"\n== {total - failures}/{total} 一致 ==")
    if failures:
        print(
            "判定が変わっています。意図した変更なら CASES の期待値を直し、"
            "そうでなければ fact_extract.py の語尾・時間表現の規則を見直してください。"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
