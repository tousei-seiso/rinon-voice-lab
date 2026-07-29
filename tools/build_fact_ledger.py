"""既存の会話ログ（memory.sqlite3）から事実台帳（facts）を一括構築する。

台帳は「誰が・誰に・何を・どうした」を正規化して持つテーブルで、列挙質問
（「作ってあげた料理を全部挙げて」）と時系列質問（「一番最初に買ってあげた本は？」）を
検索ではなく **集計** で answering するための索引。cosine top-k は該当が k 件を
超えたら構造的に溢れるため、網羅を保証するにはこの台帳が要る。

抽出はハイブリッド（fact_extract を参照）:
  1) 授受表現（〜してあげた／くれた）＋ role のルールで主客を確定（LLM 呼び出しゼロ）
  2) ルールで決まらなかった往復だけをローカル LLM へ（LM Studio）
``--rule-only`` を付けると 2) を行わない（LM Studio が落ちていても走る／完全に無料）。

使い方（app.py を動かす Python 環境 / .venv 有効、リポジトリ直下から）:
  python tools/build_fact_ledger.py --dry-run                    # 抽出結果を見るだけ
  python tools/build_fact_ledger.py --char ruri                  # 特定キャラを構築
  python tools/build_fact_ledger.py --char ruri --rule-only      # LLM を使わない
  python tools/build_fact_ledger.py --char ruri --limit 50       # 先頭50往復だけ試す
  python tools/build_fact_ledger.py --char ruri --reset          # 台帳を作り直す
  python tools/build_fact_ledger.py --char ruri --show           # 構築後に一覧を表示

途中で止めても、次回は「まだ抽出していない往復」から再開する
（facts.source_id の有無で判定するため、何度実行しても重複しない）。

注意:
  - 主客の判定にはユーザー名とキャラクター名が必要。省略時は
    profiles/latest_session.json と profiles/characters.json から推定する。
    ここを間違えると台帳の主客が全部ズレるので、--show で必ず目視確認すること。
  - 本ツールは memories を読むだけで、会話ログ自体は一切変更しない。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import fact_extract as fx  # noqa: E402
import rag_memory as rag  # noqa: E402

SESSION_ROOT = _ROOT / "profiles" / "sessions"
SESSION_PROFILE = _ROOT / "profiles" / "latest_session.json"
CHARACTER_PROFILE = _ROOT / "profiles" / "characters.json"


def _target_char_ids(only: list[str]) -> list[str]:
    if not SESSION_ROOT.exists():
        return []
    wanted = {str(c).strip() for c in only if str(c).strip()} or None
    ids: list[str] = []
    for char_dir in sorted(SESSION_ROOT.iterdir()):
        if not char_dir.is_dir() or not (char_dir / "memory.sqlite3").exists():
            continue
        if wanted is not None and char_dir.name not in wanted:
            continue
        ids.append(char_dir.name)
    return ids


def _default_user_name() -> str:
    """latest_session.json の userAddress をユーザー名として使う。"""
    try:
        data = json.loads(SESSION_PROFILE.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return ""
    settings = data.get("settings") if isinstance(data, dict) else None
    if isinstance(settings, dict):
        return str(settings.get("userAddress") or "").strip()
    return ""


def _char_name_map() -> dict[str, str]:
    """characters.json から charId -> 表示名の対応を作る。"""
    try:
        data = json.loads(CHARACTER_PROFILE.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    characters = data.get("characters") if isinstance(data, dict) else None
    result: dict[str, str] = {}
    if isinstance(characters, list):
        for item in characters:
            if isinstance(item, dict) and item.get("id"):
                result[str(item["id"])] = str(item.get("name") or item["id"]).strip()
    return result


def _make_llm(model: str, generation_mode: str):
    """LM Studio へ抽出リクエストを投げる callable を作る（app.py の実装を再利用）。"""
    import app  # 遅延 import: --rule-only では LM Studio 設定を読み込む必要が無い

    return app.make_fact_llm(model or None, generation_mode)


def build(
    char_ids: list[str],
    *,
    dry_run: bool,
    reset: bool,
    rule_only: bool,
    limit: int,
    user_name: str,
    char_name: str,
    model: str,
    generation_mode: str,
    show: bool,
) -> None:
    names = _char_name_map()
    fallback_user = user_name or _default_user_name()
    llm = None if rule_only else _make_llm(model, generation_mode)
    grand = {"turns": 0, "facts": 0, "llm": 0, "empty": 0}
    for char_id in char_ids:
        who = char_name or names.get(char_id) or char_id
        if reset and not dry_run:
            removed = rag.clear_facts(char_id)
            print(f"[{char_id}] 台帳を消去: {removed} 行")
        turns = rag.iter_unextracted_turns(char_id, limit=limit)
        print(
            f"\n[{char_id}] 未抽出の往復: {len(turns)} 件"
            f"（ユーザー名={fallback_user or '(未設定)'} / キャラ名={who}）"
        )
        counts = {"turns": 0, "facts": 0, "llm": 0, "empty": 0}
        started = time.time()
        for index, turn in enumerate(turns, start=1):
            rule_facts = fx.extract_rule_based(
                turn["user_text"],
                turn["reply_text"],
                user_name=fallback_user,
                char_name=who,
                mode=turn["mode"],
                speaker=turn["speaker"],
            )
            used_llm = llm is not None and fx.needs_llm(rule_facts)
            facts = fx.extract(
                turn["user_text"],
                turn["reply_text"],
                user_name=fallback_user,
                char_name=who,
                mode=turn["mode"],
                speaker=turn["speaker"],
                llm=llm,
            )
            counts["turns"] += 1
            if used_llm:
                counts["llm"] += 1
            if not facts:
                counts["empty"] += 1
            if dry_run:
                if facts:
                    head = " / ".join(
                        f"{f.get('subject') or '?'}→{f.get('recipient') or '?'}"
                        f" {f.get('verb')}:{f.get('object')}[{f.get('direction')}]"
                        for f in facts[:3]
                    )
                    print(f"  {rag.short_date(turn['ts']) or '----------'} {head}")
                continue
            saved = rag.save_facts(
                char_id,
                facts,
                source_id=turn["id"],
                ts=turn["ts"],
                slot=turn["slot"],
                mode=turn["mode"],
            )
            counts["facts"] += saved
            # 進捗（LLM 抽出が混ざると時間がかかるので定期的に出す）。
            if index % 25 == 0 or index == len(turns):
                elapsed = time.time() - started
                print(
                    f"  {index}/{len(turns)} 往復 / 事実 {counts['facts']} 件 / "
                    f"LLM {counts['llm']} 回 / {elapsed:.1f}s"
                )
        for key in grand:
            grand[key] += counts[key]
        stats = rag.facts_stats(char_id)
        print(
            f"[{char_id}] 完了: 往復={counts['turns']} 追加した事実={counts['facts']} "
            f"LLM使用={counts['llm']} 事実なし={counts['empty']} / 台帳合計={stats['count']} "
            f"（方向別: {stats['directions']}）"
        )
        if show and not dry_run:
            print(rag.build_ledger_block(rag.query_facts(char_id, limit=40)) or "  (空)")
    print(
        f"\n== 台帳構築{'(dry-run)' if dry_run else '完了'}: "
        f"往復={grand['turns']} 事実={grand['facts']} LLM={grand['llm']} =="
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--char", action="append", default=[], help="対象キャラID（複数可）")
    parser.add_argument("--dry-run", action="store_true", help="保存せず抽出結果だけ表示")
    parser.add_argument("--reset", action="store_true", help="既存の台帳を消してから構築")
    parser.add_argument("--rule-only", action="store_true", help="LLM 抽出を使わない")
    parser.add_argument("--limit", type=int, default=0, help="処理する往復数の上限")
    parser.add_argument("--user-name", default="", help="ユーザーの呼び名（主客判定に使う）")
    parser.add_argument("--char-name", default="", help="キャラクター名（主客判定に使う）")
    parser.add_argument("--model", default="", help="抽出に使う LM Studio モデル")
    parser.add_argument(
        "--generation-mode", default="prefill", help="抽出時の生成モード（既定 prefill＝高速）"
    )
    parser.add_argument("--show", action="store_true", help="構築後に台帳の一覧を表示")
    args = parser.parse_args()

    if not args.rule_only and not rag.is_ready():
        # LLM 抽出自体は埋め込み不要だが、保存（save_facts→_connect）は DB を触るので
        # rag_memory が無効化されていないかは確認しておく。
        print("警告: rag_memory の埋め込みバックエンドが無効です（台帳の保存自体は可能）。")
    char_ids = _target_char_ids(args.char)
    if not char_ids:
        print("対象キャラが見つかりません（profiles/sessions/<charId>/memory.sqlite3 が必要）。")
        return
    build(
        char_ids,
        dry_run=args.dry_run,
        reset=args.reset,
        rule_only=args.rule_only,
        limit=args.limit,
        user_name=args.user_name,
        char_name=args.char_name,
        model=args.model,
        generation_mode=args.generation_mode,
        show=args.show,
    )


if __name__ == "__main__":
    main()
