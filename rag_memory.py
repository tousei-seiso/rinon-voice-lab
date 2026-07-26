"""ローカル完全CPU・VRAM0 の RAG 長期記憶レイヤー。

設計（B案: 軽量案）:
  - 埋め込み: fastembed(ONNX Runtime) の ``intfloat/multilingual-e5-small``。
    PyTorch を使わず CPU 実行するため、GPU VRAM を一切消費しない。
  - ベクトル保存: Python 標準ライブラリ ``sqlite3``（キャラクターごとに 1 ファイル）。
    ``chromadb`` / ``langchain`` を持ち込まず、本体の stdlib 中心設計を保つ。
  - 類似検索: numpy による総当たり cosine 類似度 top-k。
    個人のキャラ会話ログ規模（数千〜数万往復）では総当たりで十分高速。

依存（fastembed / numpy）が未インストールでも本モジュールの import は成功し、
機能は自動的に無効化（フォールバック）される。app.py 本体の
「サードパーティ依存ゼロで起動する」動作を壊さないための設計。

e5 系モデルは埋め込み時に接頭辞が必須:
  - 保存する過去ログ: ``passage: <本文>``
  - 検索クエリ:       ``query: <本文>``
本モジュールが必ず付与するため、呼び出し側は素のテキストを渡せばよい。
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from pathlib import Path

# --- 設定（環境変数で上書き可能）--------------------------------------------
RAG_ENABLED = os.environ.get("RAG_MEMORY_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
    "",
}
_EMBED_MODEL_NAME = os.environ.get("RAG_EMBED_MODEL", "intfloat/multilingual-e5-small")
# multilingual-e5-small の埋め込み次元。
_EMBED_DIM = int(os.environ.get("RAG_EMBED_DIM", "384"))
# 既定で差し込む過去ログの件数。
# 注意: 既定 3 は小さすぎた。e5 はスコアが高域に圧縮され、感情的で長い往復が上位を
# 占めやすいため、事実（作った料理など）を述べた往復はランキング 4〜8 位に沈みがち。
# 実測（ruri 503件）で「作った料理」系の記憶は 4〜6 位に居り、k=3 では全て足切りされて
# いた。列挙・想起の取りこぼしを防ぐため既定を 8 にする。文脈長／速度・オフトピックな
# 感情記憶の混入が気になる環境は RAG_TOP_K で下げられる（ローカル LM なので API 課金は
# 増えず、効くのは context 圧迫と速度のみ。実際に入る件数は min_score による上限内）。
DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "8"))
# cosine 類似度の下限。これ未満の記憶は「関係が薄い」として差し込まない。
# 注意: multilingual-e5-small は無関係な文でも 0.75〜0.82 程度の高い cosine を
# 返す（スコア分布が高域に圧縮される）モデル。実測では、明確に関係する料理の記憶
# ですら 0.79 前後になり、旧既定 0.80 だと本当に関係する記憶まで足切りされていた。
# しかも recall は降順走査で「最初に閾値を割った時点で break」するため、途中に
# 0.80 未満の 1 件があると、それ以降の関連記憶がまとめて捨てられていた。
# → 取りこぼしを防ぐため 0.75 を既定にする。ノイズ混入は top-k と LLM 側の
#    織り込み指示で抑える。RAG_MIN_SCORE を上げると想起率は下がる（本末転倒）ので注意。
DEFAULT_MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.75"))
# 1 レコードあたりのプロンプト表示上限（文脈肥大の防止）。
_SNIPPET_LIMIT = int(os.environ.get("RAG_SNIPPET_LIMIT", "140"))

# 会話ログと同じ profiles/sessions/<charId>/ 配下に memory.sqlite3 を置く。
_APP_ROOT = Path(__file__).resolve().parent
_SESSION_ROOT = _APP_ROOT / "profiles" / "sessions"

# 埋め込みバックエンドは初回のみ遅延ロードするシングルトン。
# _backend: "fastembed" | "torch" | None
_backend = None
_backend_ready = False
_model = None       # fastembed: TextEmbedding / torch: transformers の AutoModel
_tokenizer = None   # torch バックエンドのトークナイザ
_model_error = ""
_model_lock = threading.Lock()


def _safe_id(value: object, fallback: str = "rinon") -> str:
    """キャラクター ID をフォルダ名に使える安全な文字列へ整形する（app 側と同一規則）。"""
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", str(value or "").strip()).strip("_")
    return cleaned[:64] or fallback


def _load_backend() -> None:
    """埋め込みバックエンドを 1 度だけロードする（_model_lock 保持前提）。

    1) fastembed(ONNX, 軽量) を優先。無ければ
    2) transformers + torch(CPU 固定) にフォールバック。
       TTS 環境のように torch が既に入っている場合、追加依存ゼロ・protobuf 競合
       なしで動く（fastembed の onnxruntime は protobuf>=4.25 を要求し TTS と衝突する）。
    どちらも不可なら _backend=None（機能は自動フォールバックで無効化）。
    """
    global _backend, _backend_ready, _model, _tokenizer, _model_error
    errors: list[str] = []
    # 1) fastembed（ONNX / CPU）
    try:
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=_EMBED_MODEL_NAME)
        _backend = "fastembed"
        _backend_ready = True
        return
    except Exception as exc:  # 未導入・初回DL失敗・依存衝突 など
        errors.append(f"fastembed: {type(exc).__name__}: {exc}")
    # 2) transformers + torch（CPU 固定 → VRAM 不使用）
    try:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(_EMBED_MODEL_NAME)
        model = AutoModel.from_pretrained(_EMBED_MODEL_NAME)
        model.eval()
        model.to("cpu")  # GPU/VRAM を使わない
        _model = model
        _backend = "torch"
        _backend_ready = True
        return
    except Exception as exc:
        errors.append(f"torch/transformers: {type(exc).__name__}: {exc}")
    _backend = None
    _backend_ready = True
    _model_error = " | ".join(errors)


def _get_backend():
    """ロード済みバックエンド名（"fastembed"/"torch"）を返す。未ロードならロード。失敗時 None。"""
    if _backend_ready:
        return _backend
    with _model_lock:
        if not _backend_ready:
            _load_backend()
    return _backend


def is_ready() -> bool:
    """埋め込みバックエンドが利用可能かどうか（診断用）。"""
    return RAG_ENABLED and _get_backend() is not None


def status() -> dict:
    """現在の RAG 記憶レイヤーの状態を返す（起動ログ / 診断用）。"""
    backend = _get_backend() if RAG_ENABLED else None
    return {
        "enabled": bool(RAG_ENABLED),
        "ready": bool(backend is not None),
        "backend": backend or "",
        "model": _EMBED_MODEL_NAME,
        "dim": _EMBED_DIM,
        "topK": DEFAULT_TOP_K,
        "minScore": DEFAULT_MIN_SCORE,
        "error": _model_error,
    }


def _torch_embed(prepared_texts: list[str]):
    """transformers+torch で e5 埋め込み（平均プーリング＋L2正規化, CPU）。"""
    import numpy as np
    import torch

    inputs = _tokenizer(
        prepared_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = _model(**inputs)
    token_emb = outputs.last_hidden_state  # (B, T, H)
    mask = inputs["attention_mask"].unsqueeze(-1).to(token_emb.dtype)  # (B, T, 1)
    summed = (token_emb * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    mean = summed / counts
    mean = torch.nn.functional.normalize(mean, p=2, dim=1)
    return mean.cpu().numpy().astype("float32")


def _embed_batch(texts: list[str], *, kind: str):
    """複数テキストを L2 正規化済みベクトル列へ変換する。失敗時 None。

    kind='passage'（保存する過去ログ） / 'query'（検索クエリ）。e5 は接頭辞が
    必須のため、ここで必ず付与する。バックエンドが変わっても同一次元・同一意味空間。
    """
    backend = _get_backend()
    if backend is None:
        return None
    prefix = "query: " if kind == "query" else "passage: "
    prepared = [prefix + str(t or "").strip() for t in texts]
    try:
        import numpy as np

        if backend == "fastembed":
            result = []
            for vec in _model.embed(prepared):
                v = np.asarray(vec, dtype="float32")
                norm = float(np.linalg.norm(v))
                result.append(v / norm if norm > 0.0 else v)
            return result
        # torch バックエンド（既に L2 正規化済み）
        matrix = _torch_embed(prepared)
        return [matrix[i] for i in range(matrix.shape[0])]
    except Exception:
        return None


def _embed_one(text: str, *, kind: str):
    """テキスト 1 件を L2 正規化済みベクトルへ変換する。失敗時 None。"""
    body = str(text or "").strip()
    if not body:
        return None
    vectors = _embed_batch([body], kind=kind)
    if not vectors:
        return None
    return vectors[0]


def _db_path(char_id: str) -> Path:
    return _SESSION_ROOT / _safe_id(char_id) / "memory.sqlite3"


def _connect(char_id: str, *, create: bool) -> sqlite3.Connection:
    path = _db_path(char_id)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    # 同時アクセス時の "database is locked" を緩和する。
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memories ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "slot TEXT NOT NULL DEFAULT 'main', "
        "ts TEXT NOT NULL, "
        "user_text TEXT NOT NULL, "
        "reply_text TEXT NOT NULL, "
        "mode TEXT NOT NULL DEFAULT 'normal', "
        "speaker TEXT NOT NULL DEFAULT '', "
        "embedding BLOB NOT NULL)"
    )
    # 旧スキーマで作られた DB への後方互換移行（列が無ければ追加、あれば無視）。
    for column, decl in (
        ("mode", "TEXT NOT NULL DEFAULT 'normal'"),
        ("speaker", "TEXT NOT NULL DEFAULT ''"),
    ):
        try:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {column} {decl}")
        except sqlite3.OperationalError:
            pass  # 既に存在する
    return conn


def save_memory(
    char_id: str,
    slot: str,
    user_text: str,
    reply_text: str,
    mode: str = "normal",
    speaker: str = "",
    ts: str = "",
) -> bool:
    """1 往復（ユーザー発言＋返答）をベクトル化して Chroma ではなく sqlite へ保存する。

    断片化を避けるため、往復をひとつの passage としてまとめて埋め込む。
    ``mode='two_only'`` は「2人だけモード」の記録で、user_text は実発話ではなく
    お題/進行指示であることを表す（想起時のラベル出し分けに使う）。``speaker`` は
    その返答を発したキャラクター名。埋め込みや DB 書き込みに失敗しても例外は投げず
    False を返す（本処理は継続）。
    """
    if not RAG_ENABLED:
        return False
    user_text = str(user_text or "").strip()
    reply_text = str(reply_text or "").strip()
    if not user_text or not reply_text:
        return False
    mode = "two_only" if str(mode or "").strip() == "two_only" else "normal"
    speaker = str(speaker or "").strip()
    # 元会話の時刻を渡せる（backfill/再構築で使う）。未指定なら保存時刻を刻む。
    ts = str(ts or "").strip() or time.strftime("%Y-%m-%dT%H:%M:%S%z")
    # 埋め込み本文のラベルも世界観に合わせる（2人だけモードはユーザー概念が無い）。
    stimulus_label = "お題" if mode == "two_only" else "ユーザー"
    reply_label = speaker if (mode == "two_only" and speaker) else "返答"
    passage = f"{stimulus_label}: {user_text}\n{reply_label}: {reply_text}"
    embedding = _embed_one(passage, kind="passage")
    if embedding is None:
        return False
    try:
        conn = _connect(char_id, create=True)
        try:
            conn.execute(
                "INSERT INTO memories (slot, ts, user_text, reply_text, mode, speaker, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(slot or "main"),
                    ts,
                    user_text,
                    reply_text,
                    mode,
                    speaker,
                    embedding.tobytes(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return True
    except Exception:
        return False


def recall_memory(
    char_id: str,
    query_text: str,
    k: int = DEFAULT_TOP_K,
    *,
    slot: str | None = None,
    mode: str | None = None,
    recent_user_texts: list | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[dict]:
    """現在のユーザー発言に意味的に近い過去ログを類似度順に上位 k 件返す。

    ``slot`` を渡すとその話者スロット、``mode`` を渡すとその会話モード
    （'normal' / 'two_only'）の記憶だけに絞り込む。1P 通常会話と 2P モードで
    記憶が混ざらないよう、呼び出し側は現在のターンの mode を渡すこと。
    見つからない / DB が無い / モデル未導入 のいずれでも空リストを返す
    （呼び出し側は従来の文脈生成へフォールバックする）。
    """
    if not RAG_ENABLED:
        return []
    query_text = str(query_text or "").strip()
    if not query_text:
        return []
    query_vec = _embed_one(query_text, kind="query")
    if query_vec is None:
        return []
    path = _db_path(char_id)
    if not path.exists():
        return []
    try:
        import numpy as np

        conn = sqlite3.connect(str(path), timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            # slot / mode が指定された場合は、その話者スロット・会話モードの記憶だけを
            # 対象にする。従来は両引数を受け取りながら SQL に反映しておらず、
            #   ・1P(main)/2P(second) の記憶が混ざる（slot 未反映）
            #   ・通常会話(normal)に「2人だけモード(two_only)」のお題が混ざる（mode 未反映）
            # という取り違えが起きていた。特に mode 混在は、2P モードのお題として送った
            # 「〇〇ラーメン」等が 1P 通常会話で「ユーザーが作った料理」として誤想起される
            # 原因になっていた。1P 通常会話は normal のみ、2P モードは two_only のみを想起する。
            # （backfill は全件 slot='main' / mode='normal' で投入される点に注意）。
            conditions: list[str] = []
            params: list[str] = []
            slot_key = str(slot or "").strip()
            if slot_key:
                conditions.append("slot = ?")
                params.append(slot_key)
            mode_key = str(mode or "").strip()
            if mode_key:
                conditions.append("mode = ?")
                params.append(mode_key)
            where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
            rows = conn.execute(
                "SELECT ts, user_text, reply_text, mode, speaker, embedding FROM memories"
                + where,
                params,
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return []
    if not rows:
        return []

    recent = {
        str(text or "").strip()
        for text in (recent_user_texts or [])
        if str(text or "").strip()
    }

    vectors = []
    meta: list[tuple] = []
    for ts, user_text, reply_text, mode, speaker, blob in rows:
        # サイズ不正な行（次元変更・破損など）は安全にスキップ。
        if not blob or len(blob) != _EMBED_DIM * 4:
            continue
        vectors.append(np.frombuffer(blob, dtype="float32"))
        meta.append((ts, user_text, reply_text, mode, speaker))
    if not vectors:
        return []

    try:
        matrix = np.vstack(vectors)
        scores = matrix @ query_vec  # 各行は正規化済み → 内積 = cosine 類似度
        order = np.argsort(-scores)
    except Exception:
        return []

    results: list[dict] = []
    want = max(1, int(k or DEFAULT_TOP_K))
    for idx in order:
        score = float(scores[int(idx)])
        if score < min_score:
            break  # 以降はさらに低い（無関係）ので打ち切り。
        ts, user_text, reply_text, mode, speaker = meta[int(idx)]
        # 直近履歴に既に含まれる往復は差し込まない（重複回避）。
        if str(user_text or "").strip() in recent:
            continue
        results.append(
            {
                "ts": ts,
                "user_text": user_text,
                "reply_text": reply_text,
                "mode": mode,
                "speaker": speaker,
                "score": round(score, 4),
            }
        )
        if len(results) >= want:
            break
    return results


def _compact(value: object, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _short_date(ts: object) -> str:
    match = re.match(r"(\d{4}-\d{2}-\d{2})", str(ts or ""))
    return match.group(1) if match else ""


def build_memory_block(memories: list[dict], snippet_limit: int = _SNIPPET_LIMIT) -> str:
    """recall_memory の結果を、優先度（類似度）順に順位付けした参考テキストへ整形する。"""
    if not memories:
        return ""
    lines: list[str] = []
    for rank, mem in enumerate(memories, start=1):
        date = _short_date(mem.get("ts"))
        user_text = _compact(mem.get("user_text"), snippet_limit)
        reply_text = _compact(mem.get("reply_text"), snippet_limit)
        # 2人だけモードの記憶は「ユーザー」を出さず、お題＋話者名で表す
        # （userレス世界に幻のユーザーを差し込まないため）。
        if str(mem.get("mode") or "normal") == "two_only":
            user_label = "お題"
            reply_label = str(mem.get("speaker") or "").strip() or "相手"
        else:
            user_label = "ユーザー"
            reply_label = "返答"
        head = f"{rank}." + (f"（{date}）" if date else "")
        lines.append(f"{head} {user_label}: {user_text} / {reply_label}: {reply_text}")
    return "\n".join(lines)
