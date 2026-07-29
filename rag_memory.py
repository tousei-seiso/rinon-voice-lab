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

3 つの想起チャネル（ベクトルだけでは届かない質問に答えるため）:
  1) ベクトル（意味）: ``recall_memory``。話題の近さで上位 k 件。
  2) 語彙（全件走査）: ``search_lexical``。FTS5(trigram) + LIKE。**件数上限なし**。
     cosine top-k は「その話題に触れた最古の 1 件」も「該当する全件」も保証できない
     （e5 のスコアは 0.78〜0.88 の狭帯に潰れており順位付けがほぼ効かないうえ、
     該当が k 件を超えたら構造的に溢れる）。語彙チャネルは順位ではなく一致で拾うので、
     「一番最初に買ってくれた本」「作った料理を全部」に必要な再現率を担保する。
  3) 事実台帳（集計）: ``query_facts``。往復から抽出した主体・行為・客体・方向を
     正規化して持つ ``facts`` テーブル。列挙質問は検索ではなく SELECT DISTINCT で
     全件が返る。主体・客体を構造として保持するので「誰が誰にしたか」を取り違えない。

時系列（いつ／最初／最後）の想起:
  各レコードは会話した時刻 ``ts`` を持つ。ベクトル検索は時間を一切見ないため、
  ``recall_memory(order='oldest'|'newest', pool_k=...)`` でスコア上位プールを確保して
  から ts で並べ替える。``memory_span`` で記録の残存期間を、``build_timeline_block``
  で日付・経過期間つきの年表を返し、「記録上の最初」を「本当の最初」と断定させない
  材料も LLM に渡す。
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
# 語彙チャネル（FTS5 + LIKE）の有効フラグ。0 にするとベクトル想起だけの従来動作。
LEXICAL_ENABLED = os.environ.get("RAG_LEXICAL_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
    "",
}
# FTS5 の trigram トークナイザは 3 文字未満の語に反応しない（索引が 3 文字単位のため
# 「本」1 文字では MATCH できない）。この長さ未満の語は LIKE 側へ回す。
_FTS_MIN_LEN = 3
# 事実台帳（facts）の有効フラグ。0 にすると台帳の読み書きをせず従来動作。
LEDGER_ENABLED = os.environ.get("RAG_LEDGER_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
    "",
}

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


# --- 時刻ユーティリティ ------------------------------------------------------
# ts は "2026-07-29T20:31:04+0900" 形式（save_memory / chat.jsonl の time と同形式）。
# 日付だけ・秒欠け・タイムゾーン欠けの値も混ざりうるので、比較や並べ替えは必ず
# ts_sort_key を通してから行う（生文字列の比較は形式差で壊れる）。
_TS_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?")
# 解釈できない ts のソートキー。降順・昇順どちらでも端へ寄せ、「最初/最後」の答えに
# 壊れた行が紛れ込まないようにする（無効値を 0 埋めすると最古扱いされてしまう）。
TS_KEY_INVALID = "99999999999999"


def ts_sort_key(value: object, *, end: bool = False) -> str:
    """ts を並べ替え・範囲比較に使える 14 桁文字列（YYYYMMDDhhmmss）へ正規化する。

    タイムゾーン差（+0900 等）は無視する。記録は同一環境で刻まれるため実用上ズレない。
    日付だけの値は ``end=True`` でその日の 23:59:59 として扱う（期間の終端に使う）。
    """
    match = _TS_RE.search(str(value or ""))
    if not match:
        return TS_KEY_INVALID
    year, month, day, hour, minute, second = match.groups()
    if hour is None:
        return f"{year}{month}{day}" + ("235959" if end else "000000")
    return f"{year}{month}{day}{hour}{minute}{second or '00'}"


def ts_epoch(value: object) -> float | None:
    """ts をローカル時刻の epoch 秒へ変換する（経過期間の計算用）。失敗時 None。"""
    key = ts_sort_key(value)
    if key == TS_KEY_INVALID:
        return None
    try:
        return time.mktime(time.strptime(key, "%Y%m%d%H%M%S"))
    except (ValueError, OverflowError):
        return None


def short_date(value: object) -> str:
    """ts から YYYY-MM-DD だけを取り出す（取れなければ空文字）。"""
    match = _TS_RE.search(str(value or ""))
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}" if match else ""


def format_stamp(value: object, *, seconds: bool = False) -> str:
    """ts を "YYYY-MM-DD hh:mm" へ整形する（時刻が無ければ日付だけ）。

    ``seconds=True`` で秒まで出す。同じ分に複数の往復が並ぶ場面（診断出力で
    1 往復ずつ確認するとき）は、秒まで無いと行の区別がつかない。
    """
    match = _TS_RE.search(str(value or ""))
    if not match:
        return ""
    date = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    if match.group(4) is None:
        return date
    stamp = f"{date} {match.group(4)}:{match.group(5)}"
    if seconds:
        stamp += f":{match.group(6) or '00'}"
    return stamp


def describe_age(value: object, now: float | None = None) -> str:
    """ts が何日/何ヶ月/何年前かを日本語で返す（例: "約1年5ヶ月前"）。

    LLM は日付の差分計算を苦手とし、平然と間違えた年数を答える。「いつ？」に答えさせる
    ときは、この文字列を計算済みで渡してそのまま使わせる（自分で計算させない）。
    """
    then = ts_epoch(value)
    if then is None:
        return ""
    current = time.time() if now is None else now
    days = int((current - then) // 86400)
    if days < 0:
        return ""  # 未来（時計ズレ）は黙って空にする
    if days == 0:
        return "今日"
    if days == 1:
        return "昨日"
    if days < 30:
        return f"{days}日前"
    months = days // 30
    if months < 12:
        return f"約{months}ヶ月前"
    years = days // 365
    rest_months = (days - years * 365) // 30
    if rest_months <= 0:
        return f"約{years}年前"
    return f"約{years}年{rest_months}ヶ月前"


def _in_range(ts: object, since_key: str, until_key: str) -> bool:
    """ts が [since_key, until_key] の範囲内か（キーは ts_sort_key 済みの 14 桁）。"""
    if not since_key and not until_key:
        return True
    key = ts_sort_key(ts)
    if key == TS_KEY_INVALID:
        return False  # 期間を指定された質問に、時刻不明の記録は混ぜない
    if since_key and key < since_key:
        return False
    if until_key and key > until_key:
        return False
    return True


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
        "lexical": bool(LEXICAL_ENABLED),
        "ledger": bool(LEDGER_ENABLED),
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
    _ensure_lexical_index(conn)
    _ensure_ledger(conn)
    return conn


def _ensure_lexical_index(conn: sqlite3.Connection) -> bool:
    """語彙検索用の FTS5(trigram) 索引を用意する。使えない環境では False を返す。

    ``content='memories'`` の外部コンテンツ表として作るので本文は二重に持たない。
    memories 側の INSERT/UPDATE/DELETE はトリガで索引へ伝播させる（保存経路が
    save_memory 以外に増えても同期が崩れない）。既存 DB へ後から張る場合は
    'rebuild' で全行から索引を作り直す（数千行なら一瞬）。

    FTS5 が無い / trigram が使えない SQLite でも例外は投げず False を返し、
    語彙検索は LIKE だけで動く（機能低下のみで停止しない）。
    """
    if not LEXICAL_ENABLED:
        return False
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        ).fetchone()
        if not exists:
            conn.execute(
                "CREATE VIRTUAL TABLE memories_fts USING fts5("
                "user_text, reply_text, content='memories', content_rowid='id', "
                "tokenize='trigram')"
            )
            conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        conn.executescript(
            "CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN"
            " INSERT INTO memories_fts(rowid, user_text, reply_text)"
            " VALUES (new.id, new.user_text, new.reply_text); END;"
            "CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN"
            " INSERT INTO memories_fts(memories_fts, rowid, user_text, reply_text)"
            " VALUES('delete', old.id, old.user_text, old.reply_text); END;"
            "CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE ON memories BEGIN"
            " INSERT INTO memories_fts(memories_fts, rowid, user_text, reply_text)"
            " VALUES('delete', old.id, old.user_text, old.reply_text);"
            " INSERT INTO memories_fts(rowid, user_text, reply_text)"
            " VALUES (new.id, new.user_text, new.reply_text); END;"
        )
        conn.commit()
        return True
    except sqlite3.Error:
        return False


def _ensure_ledger(conn: sqlite3.Connection) -> bool:
    """事実台帳（facts）を用意する。作れなければ False（台帳チャネルは無効化）。

    1 行 = 1 つの事実（誰が・何を・誰に・どうした）。``source_id`` で必ず元の往復
    （memories.id）へ紐付ける。台帳は正本ではなく索引なので、プロンプトへ出すときは
    必ず原文スニペットも一緒に見せ、最終判断の根拠は原文に置く（抽出ミスに LLM が
    引きずられないため）。UNIQUE 制約は同じ往復から同じ事実を二重登録しないための
    冪等キー（``INSERT OR IGNORE`` で再実行しても増えない）。
    """
    if not LEDGER_ENABLED:
        return False
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS facts ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "source_id INTEGER NOT NULL DEFAULT 0, "
            "ts TEXT NOT NULL DEFAULT '', "
            "slot TEXT NOT NULL DEFAULT 'main', "
            "mode TEXT NOT NULL DEFAULT 'normal', "
            "category TEXT NOT NULL DEFAULT '', "
            "subject TEXT NOT NULL DEFAULT '', "
            "verb TEXT NOT NULL DEFAULT '', "
            "object TEXT NOT NULL DEFAULT '', "
            "recipient TEXT NOT NULL DEFAULT '', "
            "direction TEXT NOT NULL DEFAULT 'unknown', "
            "confidence REAL NOT NULL DEFAULT 0.0, "
            "extractor TEXT NOT NULL DEFAULT '', "
            "snippet TEXT NOT NULL DEFAULT '', "
            "UNIQUE(source_id, category, subject, verb, object, recipient))"
        )
        # 列挙質問は category/direction/verb で絞って ts 順に並べるのが基本形。
        conn.execute(
            "CREATE INDEX IF NOT EXISTS facts_lookup "
            "ON facts (category, direction, verb, ts)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS facts_source ON facts (source_id)")
        conn.commit()
        return True
    except sqlite3.Error:
        return False


def _open_readonly(char_id: str) -> sqlite3.Connection | None:
    """検索用に既存 DB を開く（スキーマ作成はしない）。DB が無ければ None。"""
    path = _db_path(char_id)
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(str(path), timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn
    except sqlite3.Error:
        return None


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name=? AND type IN ('table','view')",
            (name,),
        ).fetchone()
        return bool(row)
    except sqlite3.Error:
        return False


def save_memory(
    char_id: str,
    slot: str,
    user_text: str,
    reply_text: str,
    mode: str = "normal",
    speaker: str = "",
    ts: str = "",
) -> bool:
    """1 往復を保存する（``save_turn`` の後方互換ラッパ。成否だけを返す）。"""
    return save_turn(
        char_id,
        slot,
        user_text,
        reply_text,
        mode=mode,
        speaker=speaker,
        ts=ts,
    ) is not None


def save_turn(
    char_id: str,
    slot: str,
    user_text: str,
    reply_text: str,
    *,
    mode: str = "normal",
    speaker: str = "",
    ts: str = "",
) -> int | None:
    """1 往復（ユーザー発言＋返答）をベクトル化して sqlite へ保存し、その行 id を返す。

    断片化を避けるため、往復をひとつの passage としてまとめて埋め込む。
    ``mode='two_only'`` は「2人だけモード」の記録で、user_text は実発話ではなく
    お題/進行指示であることを表す（想起時のラベル出し分けに使う）。``speaker`` は
    その返答を発したキャラクター名。埋め込みや DB 書き込みに失敗しても例外は投げず
    None を返す（本処理は継続）。

    戻り値の id は事実台帳（facts.source_id）から往復を引くための参照に使う。
    成否だけが欲しい呼び出し側は従来どおり ``save_memory`` を使えばよい。
    """
    if not RAG_ENABLED:
        return None
    user_text = str(user_text or "").strip()
    reply_text = str(reply_text or "").strip()
    if not user_text or not reply_text:
        return None
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
        return None
    try:
        conn = _connect(char_id, create=True)
        try:
            cursor = conn.execute(
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
            row_id = int(cursor.lastrowid or 0)
        finally:
            conn.close()
        return row_id or None
    except Exception:
        return None


def recall_memory(
    char_id: str,
    query_text: str,
    k: int = DEFAULT_TOP_K,
    *,
    slot: str | None = None,
    mode: str | None = None,
    recent_user_texts: list | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    order: str = "score",
    pool_k: int | None = None,
    since: str = "",
    until: str = "",
    score_band: float = 0.0,
) -> list[dict]:
    """現在のユーザー発言に意味的に近い過去ログを上位 k 件返す。

    ``slot`` を渡すとその話者スロット、``mode`` を渡すとその会話モード
    （'normal' / 'two_only'）の記憶だけに絞り込む。1P 通常会話と 2P モードで
    記憶が混ざらないよう、呼び出し側は現在のターンの mode を渡すこと。
    見つからない / DB が無い / モデル未導入 のいずれでも空リストを返す
    （呼び出し側は従来の文脈生成へフォールバックする）。

    時系列質問（「一番最初に買ってくれた本は？」）向けの引数:
      ``order``      'score'（既定・従来動作）/ 'oldest' / 'newest'。
                     oldest/newest はスコア上位プールを確保してから ts で並べ替える。
      ``pool_k``     並べ替え前に確保する候補数（既定は k と同じ）。cosine のスコアは
                     狭帯に潰れて順位が当てにならないので、k より広く取ってから時系列で
                     選ぶ。ここを絞ると「最古の 1 件」が上位枠から溢れて取りこぼす。
      ``score_band`` プールの最高スコアから何点下までを時系列選抜の対象にするか。
                     0 なら無効。緩い閾値のまま ts 昇順に並べると、話題違いの古い記憶が
                     「最初」の座を奪うため、話題の芯だけに絞ってから時系列で選ぶための帯。
      ``since``/``until``  期間の絞り込み（'YYYY-MM-DD' 等）。「去年の夏」の質問に使う。
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
    since_key = ts_sort_key(since) if str(since or "").strip() else ""
    until_key = ts_sort_key(until, end=True) if str(until or "").strip() else ""
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
        # 期間指定つきの質問では、範囲外・時刻不明の記録はここで落とす。
        if not _in_range(ts, since_key, until_key):
            continue
        vectors.append(np.frombuffer(blob, dtype="float32"))
        meta.append((ts, user_text, reply_text, mode, speaker))
    if not vectors:
        return []

    try:
        matrix = np.vstack(vectors)
        scores = matrix @ query_vec  # 各行は正規化済み → 内積 = cosine 類似度
        ranking = np.argsort(-scores)
    except Exception:
        return []

    order_key = str(order or "score").strip().lower()
    if order_key not in {"score", "oldest", "newest"}:
        order_key = "score"
    want = max(1, int(k or DEFAULT_TOP_K))
    # 時系列で選ぶ場合は、並べ替える前の候補プールを k より広く取る（cosine の順位は
    # 当てにならないので、狭いプールで時系列選抜すると最古/最新を取りこぼす）。
    pool_size = want if order_key == "score" else max(want, int(pool_k or want))

    candidates: list[dict] = []
    for idx in ranking:
        score = float(scores[int(idx)])
        if score < min_score:
            break  # 以降はさらに低い（無関係）ので打ち切り。
        ts, user_text, reply_text, mode, speaker = meta[int(idx)]
        # 直近履歴に既に含まれる往復は差し込まない（重複回避）。
        if str(user_text or "").strip() in recent:
            continue
        candidates.append(
            {
                "ts": ts,
                "user_text": user_text,
                "reply_text": reply_text,
                "mode": mode,
                "speaker": speaker,
                "score": round(score, 4),
                "via": "vector",
            }
        )
        if len(candidates) >= pool_size:
            break

    if order_key == "score":
        return candidates[:want]
    # 話題の芯だけに絞ってから時系列で並べ替える（帯の外の古い記憶に「最初」の座を
    # 奪わせない）。candidates はスコア降順なので先頭が最高スコア。
    if score_band and score_band > 0.0 and candidates:
        floor = float(candidates[0]["score"]) - float(score_band)
        candidates = [rec for rec in candidates if float(rec["score"]) >= floor]
    candidates.sort(
        key=lambda rec: ts_sort_key(rec.get("ts")), reverse=(order_key == "newest")
    )
    return candidates[:want]


# --- 語彙チャネル（FTS5 trigram + LIKE / 件数上限なし）------------------------
# 1 文字でも検索語として意味を持つ語は漢字 1 字（本・花・猫・服 等）。ひらがな 1 字は
# 助詞・語尾になって全件マッチするだけなので落とす。
_KANJI_RE = re.compile(r"[一-鿿々-〇]")
# LIKE のワイルドカードとして解釈される文字（検索語に混ざったら literal 化する）。
_LIKE_SPECIAL = ("\\", "%", "_")


# 活用語尾で終わる語を語幹へ縮める判定用。日本語は活用するので、語彙一致は
# 表層形のままでは外れる（クエリ「買った」／本文「買ってあげた」で不一致）。
# 部分一致検索なので、語幹まで縮めておけば活用差をまとめて吸収できる。
_INFLECTED_RE = re.compile(r"(?:た|て|だ|で|ます|ました|ません|ない|なかった)$")
_TRAILING_KANA_RE = re.compile(r"[ぁ-んァ-ヴー]+$")


def stem_token(token: str) -> str:
    """活用語を語幹へ縮める（買った→買 / 作って→作 / もらった→もら）。

    名詞を壊さないよう、縮めるのは活用語尾で終わる語だけに限る
    （「星の王子さま」「うどん」「プレゼント」はそのまま）。
    """
    text = str(token or "").strip()
    if not _INFLECTED_RE.search(text):
        return text
    stem = _TRAILING_KANA_RE.sub("", text)
    if stem and (len(stem) >= 2 or _KANJI_RE.match(stem)):
        return stem  # 漢字語幹が残った: 買った→買 / 食べた→食
    # 全部かなの活用語（もらった・あげた）は末尾 2 文字だけ落として語幹に近づける。
    if len(text) >= 4:
        return text[:-2]
    return text


def normalize_keywords(keywords: object, *, stem: bool = True) -> list[str]:
    """検索語の列を語彙検索に使える形へ整える（語幹化・重複除去・短語除去・上限）。

    2 文字以上、または漢字 1 字の語だけを残す。「作った 料理 献立」のような
    空白区切りのクエリ文字列をそのまま渡してもよい（分割して扱う）。
    ``stem=False`` で語幹化を切れる（診断で表層形のまま試すとき用）。
    """
    if isinstance(keywords, str):
        raw = re.split(r"[\s、,／/]+", keywords)
    else:
        raw = list(keywords or [])
    result: list[str] = []
    seen: set[str] = set()
    for token in raw:
        text = str(token or "").strip().strip("「」『』\"'`（）()")
        if not text:
            continue
        if stem:
            text = stem_token(text)
        if not text:
            continue
        if len(text) < 2 and not _KANJI_RE.match(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= 12:
            break
    return result


def _like_pattern(token: str) -> str:
    escaped = token
    for char in _LIKE_SPECIAL:
        escaped = escaped.replace(char, "\\" + char)
    return f"%{escaped}%"


def _fts_phrase(token: str) -> str:
    """FTS5 の MATCH 用にフレーズを二重引用符で囲む（内部の " は "" へ escape）。"""
    return '"' + token.replace('"', '""') + '"'


def search_lexical(
    char_id: str,
    keywords: object,
    *,
    slot: str | None = None,
    mode: str | None = None,
    since: str = "",
    until: str = "",
    min_hits: int = 0,
    order: str = "oldest",
    limit: int = 0,
    rescore_query: str = "",
    min_score: float = 0.0,
) -> list[dict]:
    """検索語の一致で過去ログを引き、時系列順に返す（**件数上限なし**の全件走査）。

    ベクトル検索の構造的な穴を埋めるためのチャネル:
      ・「その話題に触れた最古の 1 件」は cosine top-k では保証できない（時間を見ない
        うえ、同じ話題の記録が k 件を超えたら古い 1 件から順に溢れる）。
      ・「全部挙げて」の網羅も上位 k 件では保証できない。
    語彙一致は順位ではなく一致で拾うので、``limit=0``（無制限）なら該当を取りこぼさない。

    実装は 2 経路のハイブリッド:
      ・3 文字以上の語 → FTS5(trigram) の MATCH（索引が効く）
      ・1〜2 文字の語  → LIKE の全件スキャン（trigram は 3 文字未満に反応しないため。
        「本」のような漢字 1 字が主題になる質問はこちらで拾う）
    FTS5 が使えない環境では全語を LIKE 経路で処理する（機能低下のみで停止しない）。

    ``min_hits`` は「何語以上一致した行を採用するか」（既定は語数 3 以上なら 2、
    それ未満なら 1）。漢字 1 字の語は「本」が「本気」「日本」にも一致してしまうため、
    他の語との同時一致を要求してノイズを落とす。``rescore_query`` を渡すと保存済み
    埋め込みとの cosine を各行へ付け、``min_score`` 未満を捨てる（主題違いの除去）。

    ただし cosine による足切りは「一致語が 1 語だけ」の弱い行にしか適用しない。
    語彙一致はベクトルとは独立した証拠なので、2 語以上が一致している行を cosine で
    捨てると、このチャネルを足した意味（ベクトルが取りこぼす最古の 1 件を拾う）が
    失われる。スコアが計算できなかった行（埋め込み破損・次元不一致）も落とさない。
    """
    if not RAG_ENABLED or not LEXICAL_ENABLED:
        return []
    tokens = normalize_keywords(keywords)
    if not tokens:
        return []
    conn = _open_readonly(char_id)
    if conn is None:
        return []
    order_key = str(order or "oldest").strip().lower()
    if order_key not in {"oldest", "newest", "hits"}:
        order_key = "oldest"
    since_key = ts_sort_key(since) if str(since or "").strip() else ""
    until_key = ts_sort_key(until, end=True) if str(until or "").strip() else ""
    want_hits = int(min_hits) if min_hits else (2 if len(tokens) >= 3 else 1)
    want_hits = max(1, min(want_hits, len(tokens)))

    base_conditions: list[str] = []
    base_params: list[str] = []
    slot_key = str(slot or "").strip()
    if slot_key:
        base_conditions.append("m.slot = ?")
        base_params.append(slot_key)
    mode_key = str(mode or "").strip()
    if mode_key:
        base_conditions.append("m.mode = ?")
        base_params.append(mode_key)

    fts_tokens = [t for t in tokens if len(t) >= _FTS_MIN_LEN]
    like_tokens = [t for t in tokens if len(t) < _FTS_MIN_LEN]
    use_fts = bool(fts_tokens) and _has_table(conn, "memories_fts")
    if fts_tokens and not use_fts:
        # FTS5 が無い DB（索引作成に失敗した環境）では LIKE へ回して機能を維持する。
        like_tokens = tokens

    rows: dict[int, tuple] = {}
    try:
        select_cols = (
            "m.id, m.ts, m.user_text, m.reply_text, m.mode, m.speaker, m.embedding"
        )
        if use_fts:
            # trigram の MATCH は OR で束ねる（AND だと 1 語でも表記違いがあると全滅する）。
            match_expr = " OR ".join(_fts_phrase(t) for t in fts_tokens)
            where = " AND ".join(["memories_fts MATCH ?"] + base_conditions)
            for row in conn.execute(
                f"SELECT {select_cols} FROM memories_fts "
                "JOIN memories m ON m.id = memories_fts.rowid "
                f"WHERE {where}",
                [match_expr] + base_params,
            ):
                rows[int(row[0])] = row
        for token in like_tokens:
            where = " AND ".join(
                ["(m.user_text LIKE ? ESCAPE '\\' OR m.reply_text LIKE ? ESCAPE '\\')"]
                + base_conditions
            )
            pattern = _like_pattern(token)
            for row in conn.execute(
                f"SELECT {select_cols} FROM memories m WHERE {where}",
                [pattern, pattern] + base_params,
            ):
                rows[int(row[0])] = row
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    if not rows:
        return []

    # 一致語数を数えて閾値で絞る（trigram/LIKE の両経路をまたいで同じ基準で判定する）。
    hits_query = str(rescore_query or "").strip()
    query_vec = _embed_one(hits_query, kind="query") if hits_query else None
    results: list[dict] = []
    for row_id, (_id, ts, user_text, reply_text, row_mode, speaker, blob) in rows.items():
        if not _in_range(ts, since_key, until_key):
            continue
        haystack = f"{user_text}\n{reply_text}"
        hits = sum(1 for token in tokens if token in haystack)
        if hits < want_hits:
            continue
        score = 0.0
        if query_vec is not None and blob and len(blob) == _EMBED_DIM * 4:
            try:
                import numpy as np

                score = float(np.frombuffer(blob, dtype="float32") @ query_vec)
            except Exception:
                score = 0.0
            # 一致が 1 語だけの弱い行に限り、主題違い（「本」→「本気」等の誤爆）を
            # cosine で落とす。2 語以上一致している行は語彙証拠が十分なので残す。
            if min_score and score and hits < 2 and score < float(min_score):
                continue
        results.append(
            {
                "id": row_id,
                "ts": ts,
                "user_text": user_text,
                "reply_text": reply_text,
                "mode": row_mode,
                "speaker": speaker,
                "score": round(score, 4),
                "hits": hits,
                "via": "keyword",
            }
        )
    if order_key == "hits":
        results.sort(key=lambda rec: (-int(rec["hits"]), ts_sort_key(rec.get("ts"))))
    else:
        results.sort(
            key=lambda rec: ts_sort_key(rec.get("ts")), reverse=(order_key == "newest")
        )
    if limit and limit > 0:
        return results[: int(limit)]
    return results


def fetch_turns(
    char_id: str,
    ids: list,
    *,
    slot: str | None = None,
    mode: str | None = None,
) -> list[dict]:
    """memories.id を指定して往復の原文を取り出す（台帳 → 原文の逆引き）。

    台帳（facts）は索引であって正本ではない。プロンプトへ一覧を出すときは、その
    裏付けとなる原文も必ず一緒に見せる必要がある（抽出ミスに LLM が引きずられない
    ようにするため）。また一覧と年表が違う出来事を指していると LLM が混乱するので、
    一覧に載せた事実の出典は年表にも必ず含める。
    """
    wanted = [int(value) for value in (ids or []) if str(value or "").strip().isdigit()]
    if not wanted:
        return []
    conn = _open_readonly(char_id)
    if conn is None:
        return []
    conditions = [f"id IN ({','.join('?' for _ in wanted)})"]
    params: list = list(wanted)
    slot_key = str(slot or "").strip()
    if slot_key:
        conditions.append("slot = ?")
        params.append(slot_key)
    mode_key = str(mode or "").strip()
    if mode_key:
        conditions.append("mode = ?")
        params.append(mode_key)
    try:
        rows = conn.execute(
            "SELECT id, ts, user_text, reply_text, mode, speaker FROM memories WHERE "
            + " AND ".join(conditions),
            params,
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    results = [
        {
            "id": row[0],
            "ts": row[1],
            "user_text": row[2],
            "reply_text": row[3],
            "mode": row[4],
            "speaker": row[5],
            "score": 0.0,
            "via": "ledger",
        }
        for row in rows
    ]
    results.sort(key=lambda rec: ts_sort_key(rec.get("ts")))
    return results


def memory_span(char_id: str, *, slot: str | None = None, mode: str | None = None) -> dict:
    """記録が残っている期間（最古 / 最新の ts と件数）を返す。

    「一番最初に買ってくれた本」に答えるとき、記録そのものが途中から始まっていれば、
    出せるのは「記録上の最初」にすぎない。それを「本当の最初」と断定させないための
    材料として、プロンプトへ範囲を明示するのに使う。
    """
    conn = _open_readonly(char_id)
    if conn is None:
        return {"count": 0, "oldest": "", "newest": ""}
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
    try:
        rows = conn.execute(f"SELECT ts FROM memories{where}", params).fetchall()
    except sqlite3.Error:
        return {"count": 0, "oldest": "", "newest": ""}
    finally:
        conn.close()
    # 文字列の MIN/MAX は形式差（秒欠け・TZ 欠け）で壊れるので ts_sort_key で比べる。
    stamped = [
        (ts_sort_key(row[0]), row[0]) for row in rows if ts_sort_key(row[0]) != TS_KEY_INVALID
    ]
    if not stamped:
        return {"count": len(rows), "oldest": "", "newest": ""}
    stamped.sort()
    return {"count": len(rows), "oldest": stamped[0][1], "newest": stamped[-1][1]}


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


def build_timeline_block(
    memories: list[dict],
    snippet_limit: int = _SNIPPET_LIMIT,
    *,
    span: dict | None = None,
    now: float | None = None,
) -> str:
    """想起結果を「古い順・日付つきの年表」へ整形する（時系列質問用）。

    build_memory_block（類似度順）との違いは 3 点で、いずれも時系列質問に必要な情報:
      ・必ず古い順に並べる（先頭＝最古、末尾＝最新と LLM に断言させられる）
      ・日付と経過期間を添える（LLM は日付の差分計算を間違えるので計算済みを渡す）
      ・記録の残存範囲を先頭に出す（範囲より前は「記録が無い」だけで、
        「そんな出来事は無かった」ではないことを分からせる）
    """
    if not memories:
        return ""
    current = time.time() if now is None else now
    ordered = sorted(memories, key=lambda mem: ts_sort_key(mem.get("ts")))
    lines: list[str] = []
    if span and span.get("count"):
        oldest = short_date(span.get("oldest"))
        newest = short_date(span.get("newest"))
        if oldest and newest:
            lines.append(
                f"（記録の範囲: {oldest} 〜 {newest} / 全 {span['count']} 往復"
                f" / 今日: {time.strftime('%Y-%m-%d', time.localtime(current))}）"
            )
    for rank, mem in enumerate(ordered, start=1):
        stamp = format_stamp(mem.get("ts")) or "日時不明"
        age = describe_age(mem.get("ts"), now=current)
        user_text = _compact(mem.get("user_text"), snippet_limit)
        reply_text = _compact(mem.get("reply_text"), snippet_limit)
        if str(mem.get("mode") or "normal") == "two_only":
            user_label = "お題"
            reply_label = str(mem.get("speaker") or "").strip() or "相手"
        else:
            user_label = "ユーザー"
            reply_label = "返答"
        head = f"{rank}. {stamp}" + (f"（{age}）" if age else "")
        lines.append(f"{head} {user_label}: {user_text} / {reply_label}: {reply_text}")
    return "\n".join(lines)


# --- 事実台帳（facts）--------------------------------------------------------
# direction は「行為の向き」。誰が誰に対してしたことかを構造として保持するための列で、
# 主客の取り違え（キャラが「私は料理を作っていません」と答える等）を防ぐ核。
DIRECTIONS = {"user->char", "char->user", "char->char", "self", "unknown"}


def _fact_row(fact: dict, *, source_id: int, ts: str, slot: str, mode: str) -> tuple:
    direction = str(fact.get("direction") or "unknown").strip()
    if direction not in DIRECTIONS:
        direction = "unknown"
    return (
        int(source_id or 0),
        str(ts or ""),
        str(slot or "main"),
        "two_only" if str(mode or "").strip() == "two_only" else "normal",
        str(fact.get("category") or "").strip(),
        str(fact.get("subject") or "").strip(),
        str(fact.get("verb") or "").strip(),
        str(fact.get("object") or "").strip(),
        str(fact.get("recipient") or "").strip(),
        direction,
        float(fact.get("confidence") or 0.0),
        str(fact.get("extractor") or "").strip(),
        _compact(fact.get("snippet"), 160),
    )


def save_facts(
    char_id: str,
    facts: list[dict],
    *,
    source_id: int = 0,
    ts: str = "",
    slot: str = "main",
    mode: str = "normal",
) -> int:
    """抽出した事実を台帳へ追記し、実際に増えた行数を返す。

    ``INSERT OR IGNORE`` なので同じ往復に対して何度実行しても重複しない（UNIQUE 制約が
    冪等キー）。抽出に自信が無い事実も ``subject=''`` / ``direction='unknown'`` のまま
    保存してよい。捨てると「台帳に無い＝存在しない」と誤認する新たな漏れになるため、
    不明は不明として残し、想起時に「主体不明」枠として扱う。
    """
    if not RAG_ENABLED or not LEDGER_ENABLED or not facts:
        return 0
    # 客体の無い事実は保存しない（「言う: （空）」では何があったか分からず、列挙の
    # 邪魔になるだけ）。抽出側でも落としているが、台帳の入口でも守る。
    rows = [
        _fact_row(fact, source_id=source_id, ts=ts, slot=slot, mode=mode)
        for fact in facts
        if str(fact.get("object") or "").strip()
    ]
    if not rows:
        return 0
    try:
        conn = _connect(char_id, create=True)
        try:
            before = conn.total_changes
            conn.executemany(
                "INSERT OR IGNORE INTO facts ("
                "source_id, ts, slot, mode, category, subject, verb, object, recipient, "
                "direction, confidence, extractor, snippet) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            return conn.total_changes - before
        finally:
            conn.close()
    except Exception:
        return 0


def query_facts(
    char_id: str,
    *,
    category: str = "",
    subject: str = "",
    verb: str = "",
    object_like: str = "",
    direction: str = "",
    slot: str | None = None,
    mode: str | None = None,
    since: str = "",
    until: str = "",
    order: str = "oldest",
    limit: int = 0,
    include_unknown: bool = True,
    include_uncategorized: bool = True,
) -> list[dict]:
    """台帳を引いて事実の列を返す（列挙は検索ではなく集計なので、既定で上限なし）。

    ``direction`` を指定すると行為の向きで絞る。「俺が君に作ってあげた料理」は
    ``direction='user->char'``、「君が俺に作ってくれた料理」は ``'char->user'``。
    ``include_unknown=True``（既定）なら方向が判定できなかった行も併せて返す
    （取りこぼしを作らないため。呼び出し側は ``direction`` を見て提示を分ける）。

    ``include_uncategorized=True``（既定）は category が空の行も返す。カテゴリは
    語彙辞書による推定なので、固有名詞（「星の王子さま」）は本だと判定できず空になる。
    ここで厳格に絞ると「一番最初に買ってあげた本」の正解が落ちて、2 番目の記録が
    「最初」として答えられてしまう。カテゴリは絞り込みの補助であって条件ではない。
    """
    if not RAG_ENABLED or not LEDGER_ENABLED:
        return []
    conn = _open_readonly(char_id)
    if conn is None:
        return []
    if not _has_table(conn, "facts"):
        conn.close()
        return []
    conditions: list[str] = []
    params: list[str] = []
    category_key = str(category or "").strip()
    if category_key:
        if include_uncategorized:
            conditions.append("(category = ? OR category = '')")
        else:
            conditions.append("category = ?")
        params.append(category_key)
    for column, value in (
        ("subject", subject),
        ("verb", verb),
        ("slot", slot),
        ("mode", mode),
    ):
        text = str(value or "").strip()
        if text:
            conditions.append(f"{column} = ?")
            params.append(text)
    obj = str(object_like or "").strip()
    if obj:
        conditions.append("object LIKE ? ESCAPE '\\'")
        params.append(_like_pattern(obj))
    direction_key = str(direction or "").strip()
    if direction_key:
        if include_unknown:
            conditions.append("direction IN (?, 'unknown')")
        else:
            conditions.append("direction = ?")
        params.append(direction_key)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    try:
        rows = conn.execute(
            "SELECT id, source_id, ts, slot, mode, category, subject, verb, object, "
            f"recipient, direction, confidence, extractor, snippet FROM facts{where}",
            params,
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    since_key = ts_sort_key(since) if str(since or "").strip() else ""
    until_key = ts_sort_key(until, end=True) if str(until or "").strip() else ""
    columns = (
        "id",
        "source_id",
        "ts",
        "slot",
        "mode",
        "category",
        "subject",
        "verb",
        "object",
        "recipient",
        "direction",
        "confidence",
        "extractor",
        "snippet",
    )
    results = [
        dict(zip(columns, row)) for row in rows if _in_range(row[2], since_key, until_key)
    ]
    results.sort(
        key=lambda rec: ts_sort_key(rec.get("ts")),
        reverse=(str(order or "oldest").strip().lower() == "newest"),
    )
    if limit and limit > 0:
        return results[: int(limit)]
    return results


def facts_stats(char_id: str) -> dict:
    """台帳の件数内訳（全体 / 抽出済みソース数 / 方向別）を返す（診断・起動ログ用）。"""
    conn = _open_readonly(char_id)
    if conn is None or not _has_table(conn, "facts"):
        if conn is not None:
            conn.close()
        return {"count": 0, "sources": 0, "directions": {}}
    try:
        count = int(conn.execute("SELECT count(*) FROM facts").fetchone()[0])
        sources = int(
            conn.execute("SELECT count(DISTINCT source_id) FROM facts").fetchone()[0]
        )
        directions = {
            str(row[0]): int(row[1])
            for row in conn.execute(
                "SELECT direction, count(*) FROM facts GROUP BY direction"
            )
        }
        return {"count": count, "sources": sources, "directions": directions}
    except sqlite3.Error:
        return {"count": 0, "sources": 0, "directions": {}}
    finally:
        conn.close()


def clear_facts(char_id: str) -> int:
    """台帳を空にする（抽出のやり直し用）。削除した行数を返す。"""
    if not LEDGER_ENABLED:
        return 0
    try:
        conn = _connect(char_id, create=True)
        try:
            before = conn.total_changes
            conn.execute("DELETE FROM facts")
            conn.commit()
            return conn.total_changes - before
        finally:
            conn.close()
    except Exception:
        return 0


def iter_unextracted_turns(
    char_id: str,
    *,
    slot: str | None = None,
    mode: str | None = None,
    limit: int = 0,
) -> list[dict]:
    """まだ台帳へ抽出していない往復を古い順に返す（バッチ抽出の入力）。

    抽出済みかどうかは facts.source_id の有無で判定する。中断・追記しても続きから
    再開できるので、数千往復を数回に分けて処理できる。
    """
    conn = _open_readonly(char_id)
    if conn is None:
        return []
    conditions: list[str] = []
    params: list[str] = []
    slot_key = str(slot or "").strip()
    if slot_key:
        conditions.append("m.slot = ?")
        params.append(slot_key)
    mode_key = str(mode or "").strip()
    if mode_key:
        conditions.append("m.mode = ?")
        params.append(mode_key)
    if _has_table(conn, "facts"):
        conditions.append("m.id NOT IN (SELECT source_id FROM facts)")
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    try:
        rows = conn.execute(
            "SELECT m.id, m.ts, m.slot, m.mode, m.speaker, m.user_text, m.reply_text "
            f"FROM memories m{where}",
            params,
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    columns = ("id", "ts", "slot", "mode", "speaker", "user_text", "reply_text")
    turns = [dict(zip(columns, row)) for row in rows]
    turns.sort(key=lambda rec: ts_sort_key(rec.get("ts")))
    if limit and limit > 0:
        return turns[: int(limit)]
    return turns


# --- 同期・保守（履歴の手修正／UI からの削除に追従するための API）---------------
# 会話ログの正本は history.json（と chat.jsonl）で、memory.sqlite3 はそこから作る索引。
# 正本を編集したら索引を合わせ直す必要があるが、全再構築は全往復の埋め込みを計算し直す
# ため重い。以下は差分だけを直すための最小の操作群（tools/sync_memory.py が使う）。


def list_turns(
    char_id: str, *, slot: str | None = None, mode: str | None = None
) -> list[dict]:
    """保存済みの往復を全件返す（同期・監査用。embedding は返さない）。"""
    conn = _open_readonly(char_id)
    if conn is None:
        return []
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
    try:
        rows = conn.execute(
            "SELECT id, ts, slot, mode, speaker, user_text, reply_text FROM memories"
            + where,
            params,
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    columns = ("id", "ts", "slot", "mode", "speaker", "user_text", "reply_text")
    turns = [dict(zip(columns, row)) for row in rows]
    turns.sort(key=lambda rec: ts_sort_key(rec.get("ts")))
    return turns


def delete_memories(char_id: str, ids: list) -> int:
    """指定した往復を削除する（紐づく事実も一緒に消す）。削除した行数を返す。

    UI から会話を削除したり history.json を手で削ったりしたとき、memories を消すだけでは
    その往復から抽出した事実が台帳に残り、想起で復活してしまう。削除は必ず対で行う。
    FTS5 索引はトリガで追従するので、ここでは触らなくてよい。
    """
    wanted = [int(value) for value in (ids or []) if str(value or "").strip().isdigit()]
    if not wanted:
        return 0
    try:
        conn = _connect(char_id, create=True)
        try:
            placeholders = ",".join("?" for _ in wanted)
            if LEDGER_ENABLED:
                conn.execute(
                    f"DELETE FROM facts WHERE source_id IN ({placeholders})", wanted
                )
            # memories には FTS5 同期トリガが付いているため、total_changes ではトリガの
            # 変更まで数えてしまう（2 件削除が 6 と表示される）。rowcount で実数を取る。
            cursor = conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})", wanted
            )
            deleted = int(cursor.rowcount or 0)
            conn.commit()
            return deleted
        finally:
            conn.close()
    except Exception:
        return 0


def prune_orphan_facts(char_id: str) -> int:
    """出典の往復が既に無い事実（孤児）を台帳から削除する。削除した行数を返す。

    memories を作り直すと id が振り直されるため、古い source_id を持つ事実は
    「存在しない往復」または「別の往復」を指すことになる。放置すると、消したはずの
    出来事が想起されたり、原文と一覧が食い違ったりする。
    """
    if not LEDGER_ENABLED:
        return 0
    try:
        conn = _connect(char_id, create=True)
        try:
            before = conn.total_changes
            conn.execute(
                "DELETE FROM facts WHERE source_id NOT IN (SELECT id FROM memories)"
            )
            removed = conn.total_changes - before
            conn.commit()
            return removed
        finally:
            conn.close()
    except Exception:
        return 0


def update_turn_ts(char_id: str, row_id: int, ts: str) -> bool:
    """往復の時刻を更新する（history.json 側で時刻を手修正した場合の追従）。

    台帳の ts も同じ値へ揃える（facts.ts は往復の時刻の写しなので、片方だけ直すと
    年表と一覧の日付が食い違う）。
    """
    stamp = str(ts or "").strip()
    if not stamp:
        return False
    try:
        conn = _connect(char_id, create=True)
        try:
            conn.execute("UPDATE memories SET ts = ? WHERE id = ?", (stamp, int(row_id)))
            if LEDGER_ENABLED:
                conn.execute(
                    "UPDATE facts SET ts = ? WHERE source_id = ?", (stamp, int(row_id))
                )
            conn.commit()
            return True
        finally:
            conn.close()
    except Exception:
        return False


def rebuild_lexical_index(char_id: str) -> bool:
    """FTS5 索引を全行から作り直す（後付け・破損時の修復）。

    通常はトリガで同期されるので不要。旧 DB へ索引を張った直後や、外部ツールが
    トリガを介さず memories を書き換えた場合に使う。
    """
    if not LEXICAL_ENABLED:
        return False
    try:
        conn = _connect(char_id, create=True)
        try:
            conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            conn.commit()
            return True
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def build_ledger_block(facts: list[dict], *, now: float | None = None) -> str:
    """台帳の行を、列挙・時系列に答えさせるための一覧テキストへ整形する。

    同じ object（讃岐うどん等）が何度も出てくる場合は初回の日付を代表にし、回数を添える
    （列挙で同じ品が枠を食い潰さないため）。方向が不明な行は明示的に「（主客不明）」と
    出す。黙って混ぜると、主客を取り違えた回答の材料になる。
    """
    if not facts:
        return ""
    current = time.time() if now is None else now
    grouped: dict[tuple, dict] = {}
    for fact in facts:
        key = (
            str(fact.get("object") or "").strip(),
            str(fact.get("verb") or "").strip(),
            str(fact.get("direction") or "unknown").strip(),
        )
        entry = grouped.get(key)
        if entry is None:
            grouped[key] = {"fact": fact, "count": 1}
            continue
        entry["count"] += 1
        if ts_sort_key(fact.get("ts")) < ts_sort_key(entry["fact"].get("ts")):
            entry["fact"] = fact  # 代表は初回（最古）にする
    ordered = sorted(
        grouped.values(), key=lambda item: ts_sort_key(item["fact"].get("ts"))
    )
    lines: list[str] = []
    for rank, item in enumerate(ordered, start=1):
        fact = item["fact"]
        date = short_date(fact.get("ts")) or "日時不明"
        age = describe_age(fact.get("ts"), now=current)
        subject = str(fact.get("subject") or "").strip()
        recipient = str(fact.get("recipient") or "").strip()
        direction = str(fact.get("direction") or "unknown").strip()
        who = subject or "（主体不明）"
        to_whom = f"→{recipient}" if recipient else ""
        note = "（主客不明）" if direction == "unknown" else ""
        repeat = f" ×{item['count']}" if item["count"] > 1 else ""
        lines.append(
            f"{rank}. {date}" + (f"（{age}）" if age else "")
            + f" {who}{to_whom} が {fact.get('verb') or '?'}: "
            + f"{fact.get('object') or '?'}{repeat}{note}"
        )
    return "\n".join(lines)
