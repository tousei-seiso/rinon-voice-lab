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
    ("明日は肉じゃがを作ってあげる", ("作る", "肉じゃが", "plan", "2026-07-31")),
    # 予定の時期は未来側へ寄せる。回想と同じ規則で解くと来週の予定が去年になる
    # （実測: 2026-07-29 の「8月5日に抜糸」が 2025-08-05 と記録された）。
    ("8月5日に映画館に行こうと思ってる", ("行く", "映画館", "plan", "2026-08-05")),
    ("来月ハンバーグを作ってあげるね", ("作る", "ハンバーグ", "plan", "2026-08")),
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
    # 年の無い「3月」は、回想なら直近の過去（基準日 2026-07-30 なので今年の3月）。
    ("3月に水族館に行ったね", ("行く", "水族館", "done", "2026-03")),
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
    # --- 節の断片を客体にしない（実データの切り出し失敗）---
    ("倒れて不調なままネパールの病院に行った", ("行く", "ネパールの病院", "done", "")),
    ("薬局に行って薬を買った", ("買う", "薬", "done", "")),
    ("そのまま空港に行った", ("行く", "空港", "done", "")),
    ("また横浜に行こう", ("行く", "横浜", "plan", "")),
    # 「に」が場所を指す形は残す（目的の「に」だけを落とす。NEGATIVE_CASES と対）。
    ("薬局に行って薬を買った", ("行く", "薬局", "done", "")),
    ("鎌倉のカフェに行ったよ", ("行く", "鎌倉のカフェ", "done", "")),
    ("実家に行ってきた", ("行く", "実家", "done", "")),
    ("友達に会ったよ", ("会う", "友達", "done", "")),
)

# 抽出**されてはいけない** (発話, (動詞, 客体))。実データで台帳に入っていたゴミ。
NEGATIVE_CASES: tuple[tuple[str, tuple[str, str]], ...] = (
    # 「立ち上がって」を『登る』として拾っていた（客体は時間表現の「翌朝」）。
    ("翌朝になって何とか立ち上がって薬局に行った", ("登る", "翌朝")),
    ("薬局に行って薬を買った", ("買う", "行って薬")),
    ("倒れて不調なままネパールの病院に行った", ("行く", "倒れて不調なままネパールの病院")),
    # 連用形の切れ端（1 字の動詞語幹）と、中身の無い抽象語。
    ("後日結果を見に行ったよ", ("行く", "見")),
    ("後日結果を見に行ったよ", ("行く", "後日結果")),
    # 「〜に行く」の「に」は目的も表す。場所でない客体を「行った場所」にしない
    # （実測: 30 件中 12 件がこの形だった）。
    ("気分転換に行こうかな", ("行く", "気分転換")),
    ("また仕事に行ってくる", ("行く", "また仕事")),
    ("任務に行った", ("行く", "任務")),
    ("ドリアンの分析に行く", ("行く", "ドリアンの分析")),
    ("買い物に行った", ("行く", "買い物")),
    ("最近よく一緒に行くね", ("行く", "最近よく一緒")),
)

# 主体の判定 (発話, 動詞, 客体, 期待する主体)。'' は「主体不明のままが正しい」。
SUBJECT_CASES: tuple[tuple[str, str, str, str], ...] = (
    # 「が」は主格だけでなく動詞の一部にもなる。節の断片を主体にしない。
    ("翌朝になって何とか立ち上がって薬局に行った", "行く", "薬局", ""),
    ("盛り上がって歌を歌ったね", "歌う", "歌", ""),
    # 離れた「もらった」は別の述語のもの。向きを付けない。
    ("病院に行って薬をもらった", "行く", "病院", ""),
    # 授受表現があるなら向きから主体が決まる。
    ("ナデシコが肉じゃがを作ってくれた", "作る", "肉じゃが", "ルリ"),
    ("讃岐うどんを作ってあげたよ", "作る", "讃岐うどん", "オサム"),
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
    for text, (verb, obj) in NEGATIVE_CASES:
        facts = fx.extract_rule_based(text, "", user_name=USER, char_name=CHAR, ts=TS)
        hit = next((f for f in facts if f["verb"] == verb and f["object"] == obj), None)
        if hit is not None:
            failures += 1
            print(f"✗ {text}")
            print(f"    入ってはいけない事実が抽出された: {_describe(hit)}")
        elif args.all:
            print(f"✓ {text}\n    {verb}:{obj} は抽出されない")
    for text, verb, obj, expected_subject in SUBJECT_CASES:
        facts = fx.extract_rule_based(text, "", user_name=USER, char_name=CHAR, ts=TS)
        hit = next((f for f in facts if f["verb"] == verb and f["object"] == obj), None)
        if hit is None:
            failures += 1
            print(f"✗ {text}\n    {verb}:{obj} が抽出されなかった（主体を確認できない）")
            continue
        if hit["subject"] != expected_subject:
            failures += 1
            print(f"✗ {text}")
            print(
                f"    主体 {hit['subject'] or '(不明)'} ≠ {expected_subject or '(不明)'}"
                f" — {_describe(hit)}"
            )
        elif args.all:
            print(f"✓ {text}\n    主体={hit['subject'] or '(不明)'} {verb}:{obj}")
    for question, expected in QUESTION_CASES:
        actual = fx.infer_query_filters(question, user_name=USER, char_name=CHAR)
        if actual["modality"] != expected:
            failures += 1
            print(f"✗ 質問「{question}」→ 相={actual['modality']} ≠ {expected}")
        elif args.all:
            print(f"✓ 質問「{question}」→ 相={actual['modality']}")
    total = len(CASES) + len(NEGATIVE_CASES) + len(SUBJECT_CASES) + len(QUESTION_CASES)
    print(f"\n== {total - failures}/{total} 一致 ==")
    if failures:
        print(
            "判定が変わっています。意図した変更なら CASES の期待値を直し、"
            "そうでなければ fact_extract.py の語尾・時間表現の規則を見直してください。"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
