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

抽出の規則を直したあと、**LLM 呼び出しなしで**直せる範囲を直す（数秒で終わる）:
  # 正規表現の規則（係り受け・カテゴリ辞書・客体フィルタ）を直したとき。
  # LLM 由来の事実は文全体を読んで判断しているので影響を受けず、そのまま残る。
  python tools/build_fact_ledger.py --redo-rule
  python tools/build_fact_ledger.py --redo-rule --dry-run   # 何が入るか先に見る
  # 「同じ出来事に食い違う向き」が残っている分を畳む
  python tools/build_fact_ledger.py --fix-conflicts
  # 出来事の時期（occurred）だけを time_hint から再計算して直す
  python tools/build_fact_ledger.py --fix-occurred --dry-run
  python tools/build_fact_ledger.py --fix-occurred

抽出の規則を直したあと、**一部だけ**作り直す（全件やり直すと LLM 抽出で数十分かかる）:
  # 7月以降の往復だけをやり直す
  python tools/build_fact_ledger.py --redo --since 2026-07-01
  # 「登る」に関わる事実を含む往復だけをやり直す（動詞辞書や係り受けを直したとき）
  python tools/build_fact_ledger.py --redo-verb 登る
  # 「料理」カテゴリだけ、しかも7月以降に限る
  python tools/build_fact_ledger.py --redo-category 料理 --since 2026-07-01
  # 期間を絞って未抽出分だけ進める（やり直しはしない）
  python tools/build_fact_ledger.py --since 2026-07-20

台帳の中身だけを後から見る（抽出しない読み取り専用。**構築の途中でも使える**）:
  python tools/build_fact_ledger.py --list
  python tools/build_fact_ledger.py --list --char ruri --list-limit 200
  python tools/build_fact_ledger.py --list --char ruri --verb 作る      # 動詞で絞る
  python tools/build_fact_ledger.py --list --char ruri --category 料理  # カテゴリで絞る
  python tools/build_fact_ledger.py --list --modality plan              # 予定として記録された分だけ
  python tools/build_fact_ledger.py --list --object カルボナーラ        # あの品が台帳にあるか（部分一致）

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


class _LlmGuard:
    """LLM 呼び出しの連続失敗を数え、続きそうなら中断を知らせるラッパー。

    長時間の構築を放置して走らせるとき、LM Studio が途中で落ちても
    ``fact_extract.extract`` は例外を飲んでルール抽出の結果だけを返すため、
    **抽出は止まらず残りすべてが LLM なしの低品質な台帳になる**。
    それに気付けないのが一番困るので、連続失敗が閾値を超えたら中断する。
    """

    def __init__(self, call, limit: int = 10) -> None:
        self._call = call
        self._limit = max(1, int(limit))
        self.failures = 0
        self.total_failures = 0
        self.aborted = False
        self.last_error = ""

    def __call__(self, system: str, prompt: str) -> str:
        try:
            result = self._call(system, prompt)
        except Exception as exc:
            self.failures += 1
            self.total_failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            if self.failures >= self._limit:
                self.aborted = True
            raise
        self.failures = 0
        return result


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
    since: str = "",
    until: str = "",
    redo: bool = False,
    redo_verb: str = "",
    redo_category: str = "",
) -> None:
    names = _char_name_map()
    fallback_user = user_name or _default_user_name()
    # LM Studio が途中で落ちたら気付けるようにラップする（放置実行の保険）。
    guard = None if rule_only else _LlmGuard(_make_llm(model, generation_mode))
    llm = guard
    grand = {"turns": 0, "facts": 0, "llm": 0, "empty": 0}
    for char_id in char_ids:
        who = char_name or names.get(char_id) or char_id
        if reset and not dry_run:
            removed = rag.clear_facts(char_id)
            print(f"[{char_id}] 台帳を消去: {removed} 行")
        # 部分的な作り直し: 対象の往復の事実を先に消すと「未抽出」として拾われ、
        # 再抽出される。抽出の規則を直したときに全件やり直さずに済む。
        if (redo or redo_verb or redo_category) and not dry_run and not reset:
            targets = rag.facts_source_ids(
                char_id, verb=redo_verb, category=redo_category, since=since, until=until
            )
            if targets:
                removed = rag.delete_facts_by_sources(char_id, targets)
                label = " / ".join(
                    part
                    for part in (
                        f"動詞={redo_verb}" if redo_verb else "",
                        f"カテゴリ={redo_category}" if redo_category else "",
                        f"{since or '最初'}〜{until or '最後'}" if (since or until) else "",
                    )
                    if part
                ) or "全期間"
                print(
                    f"[{char_id}] やり直し対象（{label}）: {len(targets)} 往復 / "
                    f"事実 {removed} 件を削除しました"
                )
            else:
                print(f"[{char_id}] やり直し対象の事実はありません")
        turns = rag.iter_unextracted_turns(char_id, limit=limit, since=since, until=until)
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
                ts=turn["ts"],
            )
            used_llm = llm is not None and fx.needs_llm(rule_facts)
            facts = fx.extract(
                turn["user_text"],
                turn["reply_text"],
                user_name=fallback_user,
                char_name=who,
                mode=turn["mode"],
                speaker=turn["speaker"],
                # 回想（「1年前に行ったよね」）の出来事時期を解決する基準日。
                ts=turn["ts"],
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
                        f" {f.get('verb')}:{f.get('object')}"
                        f"[{f.get('direction')}/{f.get('modality') or 'done'}"
                        + (f"@{f.get('occurred')}" if f.get("occurred") else "")
                        + "]"
                        for f in facts[:3]
                    )
                    # 同じ日に何十往復もあるので、日付だけでは行を特定できない。
                    # 秒まで出して元の往復を追えるようにする。
                    stamp = rag.format_stamp(turn["ts"], seconds=True) or "(日時不明)"
                    print(f"  {stamp} {head}")
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
            # 進捗（LLM 抽出が混ざると時間がかかるので定期的に出す）。放置して走らせる
            # ことを想定し、経過だけでなく残り時間の目安も出す。
            if index % 25 == 0 or index == len(turns):
                elapsed = time.time() - started
                eta = (elapsed / index) * (len(turns) - index) if index else 0
                print(
                    f"  {index}/{len(turns)} 往復 / 事実 {counts['facts']} 件 / "
                    f"LLM {counts['llm']} 回 / {elapsed / 60:.1f}分経過"
                    + (f" / 残り約{eta / 60:.0f}分" if eta > 60 else "")
                    + (
                        f" / ⚠ LLM 失敗 {guard.total_failures} 回"
                        if guard is not None and guard.total_failures
                        else ""
                    ),
                    flush=True,
                )
            # LM Studio が落ちたまま走り続けると、残り全部が LLM なしの低品質な台帳に
            # なる。連続失敗が続いたら止めて、原因を直してから再開させる
            # （未抽出分から再開できるので、ここまでの成果は無駄にならない）。
            if guard is not None and guard.aborted:
                print(
                    f"\n[{char_id}] ⚠ LLM 呼び出しが {guard.failures} 回連続で失敗したため中断します。"
                    f"\n  最後のエラー: {guard.last_error}"
                    f"\n  LM Studio が起動しているか確認し、同じコマンドを --reset 無しで"
                    f"再実行してください（未抽出の往復から再開します）。",
                    flush=True,
                )
                break
        for key in grand:
            grand[key] += counts[key]
        if guard is not None and guard.aborted:
            # 次のキャラへ進んでも同じ理由で失敗するので、ここで打ち切る。
            break
        stats = rag.facts_stats(char_id)
        print(
            f"[{char_id}] 完了: 往復={counts['turns']} 追加した事実={counts['facts']} "
            f"LLM使用={counts['llm']} 事実なし={counts['empty']} / 台帳合計={stats['count']} "
            f"（方向別: {stats['directions']} / 相別: {stats['modalities']} / "
            f"出来事時刻あり: {stats['occurred']}）"
        )
        if show and not dry_run:
            list_ledger([char_id], limit=40)
    print(
        f"\n== 台帳構築{'(dry-run)' if dry_run else '完了'}: "
        f"往復={grand['turns']} 事実={grand['facts']} LLM={grand['llm']} =="
    )


def redo_rule_facts(
    char_ids: list[str],
    *,
    since: str = "",
    until: str = "",
    user_name: str = "",
    char_name: str = "",
    dry_run: bool = False,
) -> None:
    """ルール抽出由来の事実だけを作り直す（**LLM 呼び出しゼロ**）。

    係り受け・カテゴリ辞書・客体フィルタといった正規表現の規則を直したときに使う。
    LLM 由来の事実は文全体を読んで判断しているため規則変更の影響を受けないので、
    そのまま残す。全往復を LLM に投げ直すと数時間かかるが、これは数秒で終わる。

    LLM が答えを出せている往復では、ルール側の弱い事実（向きが決まらなかった行）は
    保存しない。``fact_extract.extract`` と同じ方針で、これが無いと
    「作る: 私たち」「作る: 証拠」のような切り出し失敗が LLM 由来の正しい行と並んで
    列挙に混ざる（実測でこの形が残っていた）。
    """
    names = _char_name_map()
    fallback_user = user_name or _default_user_name()
    for char_id in char_ids:
        who = char_name or names.get(char_id) or char_id
        before = rag.facts_stats(char_id)
        turns = rag.list_turns(char_id)
        if since or until:
            since_key = rag.ts_sort_key(since) if since else ""
            until_key = rag.ts_sort_key(until, end=True) if until else ""
            turns = [t for t in turns if rag._in_range(t.get("ts"), since_key, until_key)]
        print(
            f"\n[{char_id}] ルール抽出のやり直し: 対象 {len(turns)} 往復 "
            f"（現在の台帳 {before['count']} 事実 / ユーザー名={fallback_user or '(未設定)'} / キャラ名={who}）"
        )
        if dry_run:
            sample = 0
            for turn in turns:
                facts = fx.extract_rule_based(
                    turn["user_text"], turn["reply_text"], user_name=fallback_user,
                    char_name=who, mode=turn["mode"], speaker=turn["speaker"],
                    ts=turn["ts"],
                )
                if facts and sample < 10:
                    sample += 1
                    stamp = rag.format_stamp(turn["ts"], seconds=True) or "(日時不明)"
                    head = " / ".join(
                        f"{f['verb']}:{f['object']}"
                        f"[{f['direction']}/{f['modality']}"
                        + (f"@{f['occurred']}" if f["occurred"] else "")
                        + "]"
                        for f in facts[:3]
                    )
                    print(f"  {stamp} {head}")
            print("  (dry-run: 削除も保存もしていません)")
            continue
        removed = rag.delete_facts_by_extractor(char_id, "rule", since=since, until=until)
        # LLM 由来の事実が既にある往復を、その事実ごと把握しておく。
        llm_covered: dict[int, list] = {}
        for fact in rag.query_facts(char_id):
            if str(fact.get("extractor") or "") != "llm":
                continue
            llm_covered.setdefault(int(fact["source_id"]), []).append(fact)
        added = skipped = 0
        for turn in turns:
            facts = fx.extract_rule_based(
                turn["user_text"], turn["reply_text"], user_name=fallback_user,
                char_name=who, mode=turn["mode"], speaker=turn["speaker"],
                ts=turn["ts"],
            )
            covered = llm_covered.get(int(turn["id"]))
            if covered:
                # LLM が読めている往復では、向きの決まらない弱い事実と、LLM が同じ行為を
                # 返している事実は入れない（文全体を読める LLM の判断を採る）。
                kept = [
                    fact
                    for fact in facts
                    if str(fact.get("direction") or "unknown") != "unknown"
                    and not fx.covers_action(
                        covered, fact.get("verb"), fact.get("object")
                    )
                ]
                skipped += len(facts) - len(kept)
                facts = kept
            if facts:
                added += rag.save_facts(
                    char_id, facts, source_id=turn["id"], ts=turn["ts"],
                    slot=turn["slot"], mode=turn["mode"],
                )
        # 作り直したルール由来の事実が、既存の LLM 由来と食い違うことがある
        # （同じ出来事に user->char と char->user が並ぶ）。ここで必ず畳んでおく。
        conflicts = rag.resolve_ledger_conflicts(char_id)
        after = rag.facts_stats(char_id)
        print(
            f"  → ルール由来を {removed} 件削除 / {added} 件を作り直し "
            f"/ LLM が読めている分の弱い事実 {skipped} 件は入れず "
            f"/ 食い違う向きを {conflicts} 件整理 "
            f"（台帳 {before['count']} → {after['count']} 事実 / "
            f"相別 {after['modalities']}。LLM 呼び出しなし）"
        )


def fix_conflicts(char_ids: list[str]) -> None:
    """台帳に残った「同じ出来事に食い違う向き」を畳む（LLM 呼び出しなし）。"""
    for char_id in char_ids:
        before = rag.facts_stats(char_id)
        removed = rag.resolve_ledger_conflicts(char_id)
        after = rag.facts_stats(char_id)
        print(
            f"[{char_id}] 食い違う向きを整理: {removed} 件削除 "
            f"（台帳 {before['count']} → {after['count']} 事実 / 方向別 {after['directions']}）"
        )


def fix_occurred(char_ids: list[str], *, dry_run: bool = False) -> None:
    """出来事時刻（occurred）を time_hint から再計算して直す（**LLM 呼び出しゼロ**）。

    時期の解決規則を直したときに使う。原文の言い方（time_hint）は台帳に残っているので、
    往復を読み直さずに数秒で直せる（全件 LLM 再抽出なら数十分かかる）。
    実例: 未来の予定「8月5日にやる」を回想と同じ規則で解いてしまい、2026-07-29 の発話が
    2025-08-05（去年）になっていた分の修復。
    """
    for char_id in char_ids:
        facts = rag.query_facts(char_id)
        updates: list[tuple[int, str]] = []
        for fact in facts:
            hint = str(fact.get("time_hint") or "").strip()
            if not hint:
                continue  # 原文の表現が無い行は再計算のしようがない
            future = str(fact.get("modality") or "") in {"plan", "wish"}
            fixed = fx.resolve_event_time(
                hint, fx.base_date(fact.get("ts")), future=future
            )
            if fixed != str(fact.get("occurred") or ""):
                updates.append((int(fact["id"]), fixed))
        print(
            f"\n[{char_id}] 出来事時刻の再計算: 対象 {len(facts)} 事実 / "
            f"変化する行 {len(updates)} 件"
        )
        for row_id, fixed in updates[:20]:
            fact = next(f for f in facts if int(f["id"]) == row_id)
            print(
                f"  {rag.format_stamp(fact['ts']) or '(日時不明)':17} {fact['modality']:8} "
                f"{fact['verb']}: {fact['object']}"
                f" 出来事={fact['occurred'] or '(空)'} → {fixed or '(空)'}"
                f"（{fact['time_hint']}）"
            )
        if len(updates) > 20:
            print(f"  …ほか {len(updates) - 20} 件")
        if dry_run:
            print("  (dry-run: 書き込んでいません)")
            continue
        if updates:
            changed = rag.update_fact_occurred(char_id, updates)
            print(f"  → {changed} 件を更新しました（LLM 呼び出しなし）")


def list_ledger(
    char_ids: list[str],
    *,
    limit: int = 40,
    verb: str = "",
    category: str = "",
    modality: str = "",
    object_like: str = "",
) -> None:
    """台帳の中身を一覧表示する（抽出はしない読み取り専用）。

    構築の途中でも呼べる（読み取り専用で開くので、実行中の抽出を邪魔しない）。
    件数が多いので既定は先頭 limit 件。動詞・カテゴリ・相・客体で絞り込める。
    ``object_like`` は部分一致なので、「あの品は台帳に入っているのか」を直接引ける。
    """
    for char_id in char_ids:
        stats = rag.facts_stats(char_id)
        print(
            f"\n[{char_id}] 台帳={stats['count']} 事実 / 抽出済み往復={stats['sources']} "
            f"/ 方向別={stats['directions']} / 相別={stats['modalities']}"
        )
        if not stats["count"]:
            print("  (空) tools/build_fact_ledger.py で構築してください。")
            continue
        # 件数の確認と表示を分ける。絞り込みつきで limit 件だけ見て「無い」と判断すると
        # 打ち切られただけの行を見落とす（実測: plan 108 件のうち先頭 40 件しか出ず、
        # 探していた品が「入っていない」と誤読した）。
        matched = rag.query_facts(
            char_id,
            verb=verb,
            category=category,
            modality=modality,
            object_like=object_like,
        )
        facts = matched[:limit] if limit and limit > 0 else matched
        if not matched:
            print(
                "  該当なし（--verb / --category / --modality / --object の指定を"
                "確認してください）"
            )
            continue
        if verb or category or modality or object_like:
            label = " / ".join(
                part
                for part in (
                    f"動詞={verb}" if verb else "",
                    f"カテゴリ={category}" if category else "",
                    f"相={modality}" if modality else "",
                    f"客体〜{object_like}" if object_like else "",
                )
                if part
            )
            print(f"  絞り込み（{label}）: {len(matched)} 件 → 先頭 {len(facts)} 件を表示")
        # 抽出元と主客を追えるよう、日時つき・古い順で 1 行 1 事実を出す。
        for fact in facts:
            stamp = rag.format_stamp(fact["ts"], seconds=True) or "(日時不明)"
            # 相と出来事時刻も出す（予定が「した事」として入っていないかの確認に使う）。
            when = f" 出来事={fact['occurred']}" if fact["occurred"] else ""
            if fact["time_hint"]:
                when += f"（{fact['time_hint']}）"
            print(
                f"  {stamp} {fact['direction']:11} {fact['modality']:8} "
                f"{fact['subject'] or '(不明)'}→{fact['recipient'] or '(不明)'} "
                f"{fact['verb']}: {fact['object']} [{fact['category'] or '-'}]{when} "
                f"by={fact['extractor']} src={fact['source_id']}"
            )
        if len(matched) > len(facts):
            print(f"  …ほか {len(matched) - len(facts)} 件（--list-limit で増やせます）")


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
    parser.add_argument(
        "--list",
        action="store_true",
        help="抽出せず、既にある台帳の中身だけを表示する（構築の途中でも使える）",
    )
    parser.add_argument("--list-limit", type=int, default=40, help="--list で表示する件数")
    parser.add_argument("--verb", default="", help="--list の絞り込み（例: 作る）")
    parser.add_argument("--category", default="", help="--list の絞り込み（例: 料理）")
    parser.add_argument(
        "--modality",
        default="",
        help="--list の絞り込み（done=した事 / plan=予定・願望 / negated=しなかった事）",
    )
    parser.add_argument(
        "--object", default="", help="--list の絞り込み（客体の部分一致。例: カルボナーラ）"
    )
    parser.add_argument("--since", default="", help="この日付以降の往復だけを対象にする（YYYY-MM-DD）")
    parser.add_argument("--until", default="", help="この日付までの往復だけを対象にする（YYYY-MM-DD）")
    parser.add_argument(
        "--redo",
        action="store_true",
        help="抽出済みの往復もやり直す（対象範囲の事実を消してから再抽出。--since/--until で範囲指定）",
    )
    parser.add_argument(
        "--redo-verb", default="", help="この動詞の事実を含む往復だけをやり直す（例: 登る）"
    )
    parser.add_argument(
        "--redo-category", default="", help="このカテゴリの事実を含む往復だけをやり直す（例: 料理）"
    )
    parser.add_argument(
        "--redo-rule",
        action="store_true",
        help="ルール抽出由来の事実だけを作り直す（LLM 呼び出しなし・数秒。正規表現の規則を直したとき）",
    )
    parser.add_argument(
        "--fix-conflicts",
        action="store_true",
        help="台帳に残った食い違う向きを畳む（LLM 呼び出しなし・即座）",
    )
    parser.add_argument(
        "--fix-occurred",
        action="store_true",
        help="出来事時刻を time_hint から再計算して直す（LLM 呼び出しなし・数秒）",
    )
    args = parser.parse_args()

    char_ids = _target_char_ids(args.char)
    if not char_ids:
        print("対象キャラが見つかりません（profiles/sessions/<charId>/memory.sqlite3 が必要）。")
        return
    if args.list:
        # 表示だけなので抽出も埋め込みも不要。構築中でも安全に覗ける。
        list_ledger(
            char_ids,
            limit=args.list_limit,
            verb=args.verb,
            category=args.category,
            modality=args.modality,
            object_like=args.object,
        )
        return
    if args.fix_conflicts:
        fix_conflicts(char_ids)
        return
    if args.fix_occurred:
        fix_occurred(char_ids, dry_run=args.dry_run)
        return
    if args.redo_rule:
        # LLM を使わないので LM Studio は不要。数秒で終わる。
        redo_rule_facts(
            char_ids,
            since=args.since,
            until=args.until,
            user_name=args.user_name,
            char_name=args.char_name,
            dry_run=args.dry_run,
        )
        return
    if not args.rule_only and not rag.is_ready():
        # LLM 抽出自体は埋め込み不要だが、保存（save_facts→_connect）は DB を触るので
        # rag_memory が無効化されていないかは確認しておく。
        print("警告: rag_memory の埋め込みバックエンドが無効です（台帳の保存自体は可能）。")
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
        since=args.since,
        until=args.until,
        redo=args.redo,
        redo_verb=args.redo_verb,
        redo_category=args.redo_category,
    )


if __name__ == "__main__":
    main()
