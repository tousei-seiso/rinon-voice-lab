"""RAG 想起の生スコアを可視化して、想起漏れの原因を切り分ける診断ツール。

「讃岐うどん以外に作った料理を全部挙げて」のような除外・列挙質問で想起が少ないとき、
その料理が
  ・閾値の僅か下（例 0.74）に沈んでいる → min_score を下げれば拾える（C 案が効く）
  ・そもそも遠い（例 0.60）            → クエリの作り方が悪い（除外語混入/単一クエリ）
のどちらなのかを、min_score を掛けない生スコアで一覧して切り分ける。

使い方（app.py を動かす Python 環境 / .venv 有効、リポジトリ直下から）:
  python tools/diagnose_recall.py --query "讃岐うどん 以外 作った 料理"
  python tools/diagnose_recall.py --char ruri \
      --query "讃岐うどん 以外 作った 料理" --query "作った 料理 献立 夕飯 手料理"

引数を省くと、ruri/yurika の両方に対し「除外語あり」と「除外語なし」のクエリを比較する
（A 案＝除外語ドロップが効くかを一目で確認できる）。--min-score の線でどこまで拾えるかを
✓/· で示す（既定 0.75）。retrieval には一切手を入れない読み取り専用の診断。
"""

from __future__ import annotations

import argparse
import re
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

DEFAULT_CHARS = ["ruri", "yurika"]
# 除外語ありクエリ（現状の書き換え結果）と、除外語を落とした肯定クエリの比較。
DEFAULT_QUERIES = [
    "讃岐うどん 以外 作った 料理",
    "作った 料理 献立 夕飯 手料理",
]


def _compact(value: object, limit: int = 48) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _short_date(ts: object) -> str:
    match = re.match(r"(\d{4}-\d{2}-\d{2})", str(ts or ""))
    return match.group(1) if match else "----------"


def diagnose(char_id: str, query: str, k: int, min_score: float, slot: str, mode: str) -> None:
    # min_score=0.0 で閾値を掛けず、生スコアを上位 k 件そのまま覗く（想起漏れの位置を見る）。
    hits = rag.recall_memory(char_id, query, k=k, slot=slot, mode=mode, min_score=0.0)
    survivors = sum(1 for h in hits if float(h.get("score") or 0) >= min_score)
    print(f"\n[{char_id}] query={query!r}")
    print(f"  上位{len(hits)}件（min_score={min_score} なら {survivors} 件が採用）:")
    if not hits:
        print("  （ヒット無し：slot/mode の絞り込みか DB を確認）")
        return
    for rank, h in enumerate(hits, start=1):
        score = float(h.get("score") or 0)
        mark = "✓" if score >= min_score else "·"
        date = _short_date(h.get("ts"))
        user = _compact(h.get("user_text"))
        reply = _compact(h.get("reply_text"))
        print(f"  #{rank:02d} {score:.4f} {mark} ({date}) U:{user} | R:{reply}")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 想起の生スコアを一覧して想起漏れを診断")
    parser.add_argument("--char", action="append", default=[], help="char_id（既定: ruri, yurika）")
    parser.add_argument("--query", action="append", default=[], help="検索クエリ（複数可）")
    parser.add_argument("--k", type=int, default=30, help="覗く上位件数（既定 30）")
    parser.add_argument("--min-score", type=float, default=0.75, help="採用線の表示用（既定 0.75）")
    parser.add_argument("--slot", default="main", help="話者スロット（既定 main）")
    parser.add_argument("--mode", default="normal", help="会話モード normal/two_only（既定 normal）")
    args = parser.parse_args()

    if not rag.is_ready():
        st = rag.status()
        print(f"埋め込みモデルを初期化できませんでした: {st.get('error') or 'unknown'}", file=sys.stderr)
        return 2

    chars = args.char or DEFAULT_CHARS
    queries = args.query or DEFAULT_QUERIES
    for char_id in chars:
        for query in queries:
            diagnose(char_id, query, args.k, args.min_score, args.slot, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
