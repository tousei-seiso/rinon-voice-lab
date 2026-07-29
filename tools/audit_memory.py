"""記憶の取りこぼしを「保存漏れ」と「想起漏れ」に切り分ける読み取り専用の監査ツール。

列挙質問（「作った料理を全部挙げて」）で漏れが出るとき、原因は 2 種類ある。
  A. 保存漏れ  … そもそも DB にその往復が無い。何を改善しても絶対に出ない。
  B. 想起漏れ  … DB にはあるが、検索が拾えていない（top-k の窓・スコアの狭帯など）。
本ツールは A を潰すための突き合わせを行う。B は tools/diagnose_recall.py と
tools/diagnose_temporal.py で見る。

突き合わせる 3 つの数:
  ・logs/chat.jsonl の往復数（実際に交わした会話の記録。正本に最も近い）
  ・profiles/sessions/<charId>/history.json のエントリ数
  ・profiles/sessions/<charId>/memory.sqlite3 の memories 行数
加えて、時系列想起が機能する前提である **ts の健全性**（解釈できる ts の割合、
同一日時への潰れ）と、事実台帳の抽出率も確認する。

使い方（リポジトリ直下から）:
  python tools/audit_memory.py
  python tools/audit_memory.py --char ruri

chat.jsonl はキャラ別に分かれていないため、往復数はキャラ横断の合計として表示する
（キャラが 1 人なら直接比較でき、複数なら「合計が DB 合計と釣り合うか」を見る）。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import rag_memory as rag  # noqa: E402

SESSION_ROOT = _ROOT / "profiles" / "sessions"
CHAT_LOG = _ROOT / "logs" / "chat.jsonl"


def _chatlog_summary() -> dict:
    """chat.jsonl の往復数と時刻の範囲を数える。"""
    if not CHAT_LOG.exists():
        return {"turns": 0, "with_time": 0, "oldest": "", "newest": ""}
    turns = 0
    with_time = 0
    stamps: list[str] = []
    for line in CHAT_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or not record.get("reply"):
            continue
        turns += 1
        stamp = str(record.get("time") or "").strip()
        if rag.ts_sort_key(stamp) != rag.TS_KEY_INVALID:
            with_time += 1
            stamps.append(stamp)
    stamps.sort(key=rag.ts_sort_key)
    return {
        "turns": turns,
        "with_time": with_time,
        "oldest": stamps[0] if stamps else "",
        "newest": stamps[-1] if stamps else "",
    }


def _history_summary(char_id: str) -> dict:
    path = SESSION_ROOT / char_id / "history.json"
    if not path.exists():
        return {"entries": 0, "pairs": 0, "with_ts": 0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"entries": 0, "pairs": 0, "with_ts": 0}
    history = data.get("history") if isinstance(data, dict) else data
    if not isinstance(history, list):
        return {"entries": 0, "pairs": 0, "with_ts": 0}
    entries = [item for item in history if isinstance(item, dict) and item.get("content")]
    with_ts = sum(1 for item in entries if str(item.get("ts") or "").strip())
    pairs = sum(1 for item in entries if item.get("role") == "assistant")
    return {"entries": len(entries), "pairs": pairs, "with_ts": with_ts}


def _db_summary(char_id: str) -> dict:
    path = rag._db_path(char_id)
    if not path.exists():
        return {}
    try:
        conn = sqlite3.connect(str(path))
    except sqlite3.Error:
        return {}
    try:
        rows = conn.execute("SELECT slot, mode, ts FROM memories").fetchall()
        has_fts = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_fts'"
            ).fetchone()
        )
        # 出典の往復を失った事実（過去の全再構築で id がズレた／会話を消した名残）。
        # 残っていると、消したはずの出来事が想起され続ける。
        orphans = 0
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='facts'"
        ).fetchone():
            orphans = int(
                conn.execute(
                    "SELECT count(*) FROM facts "
                    "WHERE source_id NOT IN (SELECT id FROM memories)"
                ).fetchone()[0]
            )
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    valid = [row[2] for row in rows if rag.ts_sort_key(row[2]) != rag.TS_KEY_INVALID]
    day_counts = Counter(rag.short_date(ts) for ts in valid)
    return {
        "rows": len(rows),
        "slots": Counter(row[0] for row in rows),
        "modes": Counter(row[1] for row in rows),
        "with_ts": len(valid),
        "days": len(day_counts),
        "top_day": day_counts.most_common(1)[0] if day_counts else ("", 0),
        "fts": has_fts,
        "orphans": orphans,
    }


def audit(char_ids: list[str]) -> None:
    chat = _chatlog_summary()
    print(
        f"logs/chat.jsonl: {chat['turns']} 往復（時刻あり {chat['with_time']}） "
        f"{rag.short_date(chat['oldest']) or '?'} 〜 {rag.short_date(chat['newest']) or '?'}"
    )
    db_total = 0
    for char_id in char_ids:
        history = _history_summary(char_id)
        db = _db_summary(char_id)
        facts = rag.facts_stats(char_id)
        rows = db.get("rows", 0)
        db_total += rows
        print(f"\n[{char_id}]")
        print(
            f"  history.json : {history['entries']} エントリ / 返答 {history['pairs']} 件 "
            f"/ ts 付き {history['with_ts']}"
        )
        print(
            f"  memories     : {rows} 行 / ts 解釈可 {db.get('with_ts', 0)} "
            f"/ slot={dict(db.get('slots', {}))} mode={dict(db.get('modes', {}))}"
        )
        print(f"  FTS5 索引    : {'あり' if db.get('fts') else 'なし（初回保存時に作成されます）'}")
        print(
            f"  facts 台帳   : {facts['count']} 事実 / 抽出済み往復 {facts['sources']}"
            f"（未抽出 {max(0, rows - facts['sources'])}）"
        )
        print(
            f"  facts の相   : {facts['modalities'] or '(相の列なし)'}"
            f" / 出来事時刻あり {facts['occurred']}"
        )
        if facts["count"] and not facts["modalities"]:
            print(
                "  ⚠ 相（done/plan/…）の列がありません。予定・否定が「した事」として"
                "列挙に混ざります。tools/build_fact_ledger.py --redo-rule で作り直してください。"
            )
        # --- 警告 ---
        if rows and db.get("with_ts", 0) < rows:
            print(
                f"  ⚠ ts を解釈できない行が {rows - db.get('with_ts', 0)} 件あります"
                "（その行は時系列想起・期間指定の対象外になります）。"
            )
        top_day, top_count = db.get("top_day", ("", 0))
        if rows and top_count > max(20, rows * 0.5):
            print(
                f"  ⚠ {top_day} に {top_count} 行が集中しています。history.json から"
                " ts 無しで再構築した痕跡です（時系列想起は正しく動きません）。"
                " tools/rebuild_from_chatlog.py で chat.jsonl の時刻を入れ直してください。"
            )
        if history["pairs"] and rows and abs(history["pairs"] - rows) > max(5, rows * 0.1):
            print(
                f"  ⚠ history.json の返答数({history['pairs']})と memories 行数({rows})が"
                "食い違っています。保存漏れ、または 2P モードの数え方の差を確認してください。"
            )
        if rows and not facts["count"]:
            print(
                "  ⚠ 事実台帳が空です。列挙質問の網羅と主客の安定には"
                " tools/build_fact_ledger.py での構築を推奨します。"
            )
        if db.get("orphans"):
            print(
                f"  ⚠ 出典の往復を失った事実が {db['orphans']} 件あります"
                "（会話を削除した名残、または過去の再構築で id がズレた分）。"
                " 消したはずの出来事が想起され続けるので、"
                " tools/sync_memory.py --source history で掃除してください。"
            )
        if history["entries"] and rows and history["pairs"] != rows:
            print(
                "  → history.json と DB の往復数が一致していません。"
                " tools/sync_memory.py --dry-run で差分を確認できます。"
            )
    print(f"\n== memories 合計 {db_total} 行 / chat.jsonl {chat['turns']} 往復 ==")
    if chat["turns"] and db_total < chat["turns"] * 0.9:
        print(
            "⚠ DB の行数が chat.jsonl の往復数を大きく下回ります（保存漏れの疑い）。"
            " これは検索をどう改善しても答えられない種類の漏れです。"
            " tools/rebuild_from_chatlog.py での作り直しを検討してください。"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--char", action="append", default=[], help="対象キャラID（複数可）")
    args = parser.parse_args()
    wanted = {str(c).strip() for c in args.char if str(c).strip()} or None
    if not SESSION_ROOT.exists():
        print("profiles/sessions が見つかりません。")
        return
    char_ids = [
        path.name
        for path in sorted(SESSION_ROOT.iterdir())
        if path.is_dir() and (wanted is None or path.name in wanted)
    ]
    if not char_ids:
        print("対象キャラが見つかりません。")
        return
    audit(char_ids)


if __name__ == "__main__":
    main()
