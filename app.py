from __future__ import annotations

import base64
import contextlib
import gc
import json
import mimetypes
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import urllib.error
import urllib.request
import warnings
from collections import deque
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html import unescape as html_unescape
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

# 感情キャプション付き返答の整形は tools/ の再生成とも共有する（食い違うと再生成で
# 注釈文が変わってしまうため、実装は emotion_caption.py の一箇所に置く）。
from emotion_caption import build_annotated_reply


APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"
LOG_ROOT = APP_ROOT / "logs"
CHAT_LOG_PATH = LOG_ROOT / "chat.jsonl"
# 感情キャプション付きの返答テキストを記録する専用履歴（どの区間をどの感情で話したか確認用）。
EMOTION_LOG_PATH = LOG_ROOT / "chat_emotion.jsonl"
PROFILE_ROOT = APP_ROOT / "profiles"
SESSION_PROFILE_PATH = PROFILE_ROOT / "latest_session.json"
# 会話ログはキャラクターごとにフォルダ分けして保存する（profiles/sessions/<charId>/history.json）。
SESSION_HISTORY_ROOT = PROFILE_ROOT / "sessions"
CHARACTER_PROFILE_PATH = PROFILE_ROOT / "characters.json"
SAVED_AUDIO_ROOT = APP_ROOT / "saved_audio"
USER_REFERENCE_ROOT = STATIC_ROOT / "reference" / "user_refs"
LEGACY_CHARACTER_ROOT = APP_ROOT / "characters"
CHARACTER_ROOT = APP_ROOT / "Character"
DEFAULT_IRODORI_ROOT = (APP_ROOT.parent / "Irodori-TTS").resolve()
IRODORI_ROOT = Path(os.environ.get("IRODORI_ROOT", str(DEFAULT_IRODORI_ROOT))).resolve()
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234/v1").rstrip("/")

# --- 呼び出すモデルの決定 -----------------------------------------------------
# モデル名は環境変数 LM_STUDIO_MODEL から取る。ただし**モデルを差し替えると名前以外も
# 変わる**。gemma-4-12b-it@q6_k → google/gemma-4-12b-qat の入れ替えでは、思考タグが
# <think>〜</think> から <|channel>thought〜<channel|> へ変わり、名前を替えただけでは
# 思考の独白がそのまま読み上げられた。そのため動作を確かめた系列だけをカタログで受け付け、
# 対象外の名前は使わずに LM_STUDIO_DEFAULT_MODEL → LM_FALLBACK_MODEL の順へ落とす。
#
# 新しいモデルを対象にするときの手順（名前を足すだけで済むとは限らない）:
#   1) その系列が思考をどう囲むかを実際の出力で確かめる。未知の書式なら
#      LM_THINKING_FORMATS へ書式を 1 つ足す。
#   2) LM_MODEL_CATALOG へ系列を 1 エントリ足す。判定に使う語（require / exclude）と、
#      1) の書式名（thinking）を**必ず対で**書く。書式を書き忘れると起動時に落ちる。
#   3) 動かして次を確かめる。いずれも系列ごとに挙動が変わる箇所:
#      ・思考が本文へ混ざらないか（混ざれば [lm] stripped thinking markup が出る）
#      ・_thinking_prefill のプリフィルで思考を抑止できるか（テンプレートが末尾
#        assistant を受け付けないモデルでは 400 になり従来仕様へ落ちる）
#      ・response_format=json_schema を受け付けるか（LM_STRUCTURED_OUTPUT=auto なら
#        拒否を検出して自動で外す）
#      ・プロンプト末尾の /no_think のような合図が効くか（効かない系列では無害な文字列）
# 試すだけなら LM_STUDIO_ALLOWED_MODELS にカンマ区切りで並べればコードは触らずに通せる。
# ただしその場合、思考タグはカタログに載っている書式でしか落とせない。
#
# 思考タグの書式。名前を付けてここへ集め、カタログの "thinking" から参照する。
LM_THINKING_FORMATS: dict[str, tuple[str, str]] = {
    # <think> … </think>
    "xml_think": (r"<think>", r"</think>"),
    # <|channel>thought … <channel|>（表記揺れ: <|channel|> / thinking / analysis / <|/channel|>）
    "channel": (
        r"<\|?channel\|?>[ \t]*(?:thought|thinking|analysis|reasoning)",
        r"<\|?/?channel\|?>",
    ),
}
# 系列の判定は「名前に含まれる語」で行う。LM Studio は同じモデルを
#   gemma-4-12b-it / gemma-4-12b-it@q6_k /
#   lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12b-it-Q6_K.gguf
# のどの書き方でも受け取るため、末尾のファイル名だけを見ると取り違える。とくに QAT の
# ファイル名は gemma-4-12B-it-QAT-Q4_0.gguf で "it" と "qat" の両方を含むので、
# 限定の強いエントリ（QAT）を先に置き、上から順に最初に当たったものを採る。
LM_MODEL_CATALOG: tuple[dict[str, object], ...] = (
    {
        # Google 公式 QAT
        "label": "gemma-4-12b-qat",
        "require": ("gemma", "4", "12b", "qat"),
        "thinking": ("channel",),
    },
    {
        # 従来の instruction-tuned
        "label": "gemma-4-12b-it",
        "require": ("gemma", "4", "12b", "it"),
        "exclude": ("qat",),
        "thinking": ("xml_think",),
    },
)
# カタログに無い名前を一時的に通したいとき用（カンマ区切り。部分一致ではなく語の完全一致）。
LM_EXTRA_ALLOWED_MODELS = tuple(
    name.strip().lower()
    for name in os.environ.get("LM_STUDIO_ALLOWED_MODELS", "").split(",")
    if name.strip()
)
# 最後の安全弁。環境変数が両方とも未設定／空／対象外でもここへ落ちる。
LM_FALLBACK_MODEL = "gemma-4-12b-it@q6_k"


def _model_key(value: object) -> str:
    """モデル名を突き合わせ用のキーへ潰す（小文字化・``.gguf`` 除去・記号除去）。

    リポジトリ名や量子化サフィックスは**残す**。``gemma-4-12b-it@q6_k`` と
    ``…/gemma-4-12b-it-Q4_K_M.gguf`` を別物として区別したいため
    （末尾のファイル名だけを見て同一視する _model_id_matches とは目的が違う）。
    """
    text = str(value or "").strip().lower()
    if text.endswith(".gguf"):
        text = text[: -len(".gguf")]
    return re.sub(r"[^a-z0-9]", "", text)


def match_model_catalog(name: object) -> dict[str, object] | None:
    """モデル名がカタログのどの系列かを返す。対象外なら None。

    大文字小文字・前後の空白・リポジトリ名・量子化サフィックス・``.gguf`` の違いは
    「語が含まれるか」で見るので自然に吸収される。
    """
    text = str(name or "").strip().lower()
    if not text:
        return None
    for entry in LM_MODEL_CATALOG:
        exclude = entry.get("exclude") or ()
        if any(word in text for word in exclude):
            continue
        if all(word in text for word in entry["require"]):
            return entry
    if text in LM_EXTRA_ALLOWED_MODELS:
        return {"label": "LM_STUDIO_ALLOWED_MODELS", "require": ()}
    return None


def resolve_lm_model(requested: object = None, log: bool = True) -> str:
    """LM Studio へ送るモデル名を決める。

    ``requested`` → ``LM_STUDIO_MODEL`` → ``LM_STUDIO_DEFAULT_MODEL`` → ``LM_FALLBACK_MODEL``
    の順に見て、カタログに当たった最初の名前を**申告された表記のまま**返す
    （書き方を勝手に正規化しない。GGUF のフルパス指定はそれ自体が有効な指定なので、
    こちらで短い名前へ丸めると別のロード済みモデルを指してしまう恐れがある）。
    対象外の名前は使わず次の候補へ進み、どれを採ってどれを捨てたかは 1 行ログに残す
    （起動スクリプトを入れ替えたとき、意図した系列で話せているかがここで分かる）。

    ``log`` はリクエストごとの呼び出し用。False なら「採用した」ログは省き、対象外を
    捨てたときだけ残す（毎ターン同じ行が並ぶのを避けつつ、取りこぼしは見えるようにする）。
    """
    candidates = (
        ("requested", requested),
        ("LM_STUDIO_MODEL", os.environ.get("LM_STUDIO_MODEL")),
        ("LM_STUDIO_DEFAULT_MODEL", os.environ.get("LM_STUDIO_DEFAULT_MODEL")),
    )
    rejected: list[str] = []
    for source, value in candidates:
        name = str(value or "").strip()
        if not name:
            continue
        entry = match_model_catalog(name)
        if entry:
            if log or rejected:
                skipped = f" (not supported: {' / '.join(rejected)})" if rejected else ""
                print(f"[lm] model: {source}='{name}' -> {entry['label']}{skipped}")
            return name
        rejected.append(f"{source}='{name}'")
    detail = f"not supported: {' / '.join(rejected)}" if rejected else "no env value"
    print(f"[lm] model: {detail} -> fallback '{LM_FALLBACK_MODEL}'")
    return LM_FALLBACK_MODEL


# 既定モデル（起動時に 1 度だけ解決する）。リクエスト body でモデルを明示されたときは
# 画面のプルダウンで選ばれたロード済みモデルを尊重し、そちらをそのまま送る（従来どおり）。
# 画面選択もカタログで縛りたくなったら、各 payload の `model or DEFAULT_MODEL` を
# `resolve_lm_model(model)` に替えるだけでよい。
DEFAULT_MODEL = resolve_lm_model()
# ローカルRAG長期記憶レイヤー（fastembed + sqlite3, 完全CPU/VRAM0）。
# 依存(fastembed/numpy)が未導入でも import は成功し、機能は自動フォールバックされる。
try:
    import rag_memory
except Exception:  # 予期せぬ import 失敗でも本体は起動させる
    rag_memory = None
# 事実台帳の抽出器（主客ハイブリッド）。純 stdlib なので通常は失敗しないが、
# rag_memory と同じ思想で「無くても本体は動く」形にしておく。
try:
    import fact_extract
except Exception:
    fact_extract = None
DEFAULT_CONTEXT_LIMIT = int(os.environ.get("LM_STUDIO_CONTEXT_LIMIT", "8200"))
# キャラクター返答の生成待ち時間（秒）。返答が遅くて "timed out" になる場合はこの値を延ばす。
LM_STUDIO_TIMEOUT = int(os.environ.get("LM_STUDIO_TIMEOUT", "300"))
# 構造化出力（response_format=json_schema）でセグメントJSONを強制する制御。
#   auto(既定): まずスキーマ付きで送り、サーバ/モデルが拒否(HTTP 4xx)したら自動で外して以後付けない。
#   on        : 常にスキーマ付きで送る（拒否時のみ 1 リクエスト分だけ外して再送）。
#   off       : 一切付けない（従来動作）。
LM_STRUCTURED_OUTPUT = os.environ.get("LM_STRUCTURED_OUTPUT", "auto").strip().lower()
# 実行中にサーバが response_format を拒否したら True にし、auto では以後付けない（プロセス内メモ）。
_lm_structured_output_unsupported = False

# --- ターン終わりの VRAM 巻き戻し ---------------------------------------------
# GPU を掴むのは 2 プロセスある。どちらが抱えているかを取り違えると効かない対策を
# 足すことになるので、掃除のたびに nvidia-smi でプロセス別の内訳をログへ残す。
#   A) 推論サーバ（LM Studio / llama-server, 別プロセス）
#      リクエストごとに「スロット」（＝1 本のコンテキスト）を使い、応答後もプロンプトの
#      KV キャッシュを載せたまま保持する。さらに同時リクエストが来ると並列用のスロットを
#      追加確保し、空いても解放しない。ただし llama.cpp は KV バッファをモデルロード時に
#      n_ctx ぶん確保しきるので、erase で戻るのは「どこまでキャッシュ済みか」の帳簿だけで、
#      VRAM のバイト数は減らない（実測: erase 成功後も 12.6GB のまま）。
#      効くのは「そもそも 2 本目のスロットを確保させない」直列化の方。
#   B) Irodori-TTS（このプロセス内, IRODORI_MODEL_DEVICE=auto なら cuda）
#      torch のキャッシュアロケータは解放済みブロックをドライバへ返さず抱え続ける。
#      文の長さごとに違うサイズのブロックが溜まるため、返答を合成するほど増えて戻らない。
#      empty_cache() で明示的に返させる。こちらは実際に VRAM が減る。
# 対策は 4 段。いずれも「無くても動く」層に留め、非対応なら黙って諦める。
#   1) 直列化   : 補助生成が本文生成へ割り込まないようにし、スロットを増やさせない。
#   2) リクエスト: cache_prompt=false を付け、プロンプト KV を残させない。
#   3) 明示解放 : 全リクエストが捌けた時点で /slots?action=erase を叩いて捨てさせる。
#   4) 自プロセス: TTS が抱えた torch の CUDA キャッシュを empty_cache() で返す。
# LM Studio へ送るリクエストを 1 本ずつに直列化する（VRAM 目的。速度目的ではない）。
# 事実抽出はバックグラウンドスレッドから飛ぶため、切ると本文生成と重なり得る。
LM_SERIALIZE_REQUESTS = os.environ.get("LM_SERIALIZE_REQUESTS", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
    "",
}
# cache_prompt=false（llama.cpp のプロンプトキャッシュ無効化）を付ける範囲。
#   aux(既定): 補助生成（クエリ書き換え・事実抽出・要点メモ）だけに付ける。返答本文は
#             履歴の共通接頭辞を再利用できた方が速いので、そちらのキャッシュは残す。
#   all      : 返答本文にも付ける（毎回プロンプトを再評価するので遅くなる）。
#   off      : 一切付けない（従来動作）。
LM_CACHE_PROMPT = os.environ.get("LM_CACHE_PROMPT", "aux").strip().lower()
if LM_CACHE_PROMPT not in {"aux", "all", "off"}:
    LM_CACHE_PROMPT = "aux"
# サーバが cache_prompt を 400 で拒否したら True にし、以後付けない（プロセス内メモ）。
_lm_cache_prompt_unsupported = False
# KV キャッシュの明示解放をいつ行うか。
#   idle(既定): 進行中の LM リクエストが 0 になった瞬間（＝ターンの後処理まで終わった時点）。
#   each      : 1 リクエストごと（補助生成の合間も毎回解放する。最も強いが最も遅い）。
#   off       : 解放しない（従来動作）。
LM_KV_CACHE_RELEASE = os.environ.get("LM_KV_CACHE_RELEASE", "idle").strip().lower()
if LM_KV_CACHE_RELEASE not in {"idle", "each", "off"}:
    LM_KV_CACHE_RELEASE = "idle"
# 解放 API の待ち時間（秒）。会話のあとの掃除なので短く切り上げる。
LM_KV_CACHE_RELEASE_TIMEOUT = float(os.environ.get("LM_KV_CACHE_RELEASE_TIMEOUT", "5"))
# idle 判定から実際に解放するまでの待ち（秒）。返答本文と補助生成の間には TTS 合成ぶんの
# 空白が空くため、0 にすると 1 ターンで何度も解放してしまう。ここで一拍待ち、その間に
# 次のリクエストが来たら予約を取り消して数え直す（＝ターンの後処理まで終わってから 1 回）。
# 長くすると「短い間隔で連投するあいだはプロンプトキャッシュを温存する」挙動になる。
LM_KV_CACHE_RELEASE_DELAY = float(os.environ.get("LM_KV_CACHE_RELEASE_DELAY", "2"))
# /slots を持たないサーバでは 1 度試して以後黙る（毎ターン失敗ログを出さないため）。
_lm_kv_release_unsupported = False
# 自プロセス（Irodori-TTS）が抱えた torch の CUDA キャッシュを掃除ごとに返すか。
VRAM_RELEASE_TORCH = os.environ.get("VRAM_RELEASE_TORCH", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
    "",
}
# 掃除の前後で nvidia-smi を叩き、プロセス別の GPU 使用量をログへ残すか。
# 「どちらのプロセスが抱えているか」はこれを見ないと切り分けられないため既定 ON。
VRAM_MEMORY_LOG = os.environ.get("VRAM_MEMORY_LOG", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
    "",
}
# nvidia-smi が無い環境（AMD/Apple/CPU 実行）では 1 度試して以後黙る。
_vram_probe_unsupported = False
# リクエストの直列化と「いま何本走っているか」の計数。RLock は同一スレッドからの
# 入れ子（空返答のリトライ等）で自分を待たないようにするため。
_lm_request_lock = threading.RLock()
_lm_inflight_lock = threading.Lock()
_lm_inflight = 0
# 遅延解放の予約（1 本だけ持ち、新しいリクエストが来たら取り消して張り直す）。
_lm_release_lock = threading.Lock()
_lm_release_timer: threading.Timer | None = None
# LLM 生成モード（思考モデルでの空返答対策と品質/速度のトレードオフを切り替える）。
#   prefill       : アシスタント・プリフィルで思考を抑止（高速・低品質）。
#   original      : 従来仕様。reply_length 由来の max_tokens をそのまま使う（思考モデルでは空になり得る）。
#   quality_guard : プリフィル無し＋大きな max_tokens 上限（品質重視、暴走はここで打ち切り）。
#   unlimited     : プリフィル無し＋max_tokens=-1（完全品質重視・上限なし）。
LM_GENERATION_MODES = ("prefill", "original", "quality_guard", "unlimited")
DEFAULT_LM_GENERATION_MODE = os.environ.get("LM_GENERATION_MODE", "quality_guard").strip().lower()
if DEFAULT_LM_GENERATION_MODE not in LM_GENERATION_MODES:
    DEFAULT_LM_GENERATION_MODE = "quality_guard"
# quality_guard モードで使う「かなり大きな」出力上限。暴走時のみここで打ち切る安全弁。
LM_QUALITY_GUARD_MAX_TOKENS = int(os.environ.get("LM_QUALITY_GUARD_MAX_TOKENS", "8192"))
# セグメント返答用の JSON スキーマ（LM Studio の response_format=json_schema 用）。
# parse_lmstudio_segments が期待する {"segments":[{text,style,emoji}...]} を強制する。
LM_SEGMENT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "irodori_segments",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "style": {"type": "string"},
                            "emoji": {"type": "string"},
                        },
                        "required": ["text", "style", "emoji"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["segments"],
            "additionalProperties": False,
        },
    },
}
LM_COMPACT_CONTEXT_LIMIT = int(os.environ.get("LM_COMPACT_CONTEXT_LIMIT", "4200"))
LM_RECENT_MESSAGE_COUNT = int(os.environ.get("LM_RECENT_MESSAGE_COUNT", "12"))
LM_SUMMARY_CHAR_LIMIT = int(os.environ.get("LM_SUMMARY_CHAR_LIMIT", "1400"))

# --- 出力枠の確保（プロンプト予算） -------------------------------------------
# 空返答の真因は max_tokens ではなく「ロード済みモデルの文脈長」である。プロンプトが
# 文脈をほぼ埋めていると、残り枠を思考(reasoning)が食い切って本文が 0 トークンになる。
# 実測（gemma-4-12b / n_ctx 8192）: prompt 7980tok で max_tokens=-1（無制限）でも
# finish_reason=length・content 空・reasoning_content だけ 800 字が返った。max_tokens を
# 8192 にした quality_guard でも完全に同じ結果で、律速は常に文脈長側だった。
# そこで毎ターン「思考＋本文ぶんの枠」を先に確保し、はみ出す分は記憶ブロック側を削る。
# 文脈長の手動指定（0 なら LM Studio へ問い合わせて自動取得）。/api/v0 を持たない
# 他の OpenAI 互換サーバではここで教える。
LM_CONTEXT_LENGTH = int(os.environ.get("LM_CONTEXT_LENGTH", "0"))
# 思考＋本文のために必ず空けておくトークン数。実測で gemma-4 の思考は 380tok 前後、
# 返答本文（long）が 400tok 弱なので、既定 1280 はおよそ 2 倍の余裕。
LM_OUTPUT_RESERVE_TOKENS = int(os.environ.get("LM_OUTPUT_RESERVE_TOKENS", "1280"))
# 自動取得した文脈長のキャッシュ秒数（モデル入れ替え・再ロードへ追従するため短く持つ）。
LM_CONTEXT_PROBE_TTL = float(os.environ.get("LM_CONTEXT_PROBE_TTL", "60"))
_lm_context_cache: dict[str, tuple[float, int]] = {}
# 記憶ブロックが枠に収まらないとき、LLM で「要点メモ」へ圧縮するか
# （0 なら LLM を挟まず、行単位の間引きだけで収める）。
LM_MEMORY_DIGEST = os.environ.get("LM_MEMORY_DIGEST", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
    "",
}
# 要点メモ 1 回ぶんの生成上限トークン。
LM_MEMORY_DIGEST_MAXTOK = int(os.environ.get("LM_MEMORY_DIGEST_MAXTOK", "700"))
# 記憶が文脈に収まらないときの分割要約（map-reduce）の最大チャンク数。
# ここで打ち切った分はプロンプトへ載らないので、件数をログへ残す。
LM_MEMORY_DIGEST_CHUNKS = max(1, int(os.environ.get("LM_MEMORY_DIGEST_CHUNKS", "4")))
# 予算オーバーで記録を間引いたことを LLM へ伝える印（黙って消すと「無かった事」にされる）。
_MEMORY_OMITTED_MARK = "…（文脈に収まらない記録は省略）…"
WEB_SEARCH_TIMEOUT = int(os.environ.get("WEB_SEARCH_TIMEOUT", "12"))

# --- TTS 読み仮名化（英字→カナ）設定 ---
# Irodori-TTS は日本語音声モデルのため、ラテン文字（英単語）を読み上げると生成が暴走し、
# 意味不明な音声になることがある。TTS へ渡す直前だけ英字をカナへ変換し、保存・表示は原文の
# まま残す。辞書は所定フォルダのファイルを使い、pip 依存(alkana 等)は増やさない方針。
#   有効/無効: TTS_KANA_NORMALIZE = on(既定) / off
#   辞書ファイル: TTS_KANA_DICT_FILE
#                 "alkanadict.csv"(tts_dictionaries/ 配下) / "tts_dictionaries/alkanadict.csv"
#                 (プロジェクト相対) / 絶対パス のいずれも可。
#                 ";" 区切りで複数指定可（後に書いたファイルが優先）。
#   形式: 1行 "english,カタカナ"（カンマ or タブ区切り／# はコメント／英字は大小無視）。
#         alkana の外部データ CSV と同一形式（alkanadict.csv を置いて指定できる）。
#   辞書が無くても、未知語はカナへフォールバックし、最終ガードでラテン文字を必ず除去する。
TTS_KANA_NORMALIZE = os.environ.get("TTS_KANA_NORMALIZE", "on").strip().lower()
TTS_DICT_ROOT = APP_ROOT / "tts_dictionaries"
TTS_KANA_DICT_FILE = os.environ.get("TTS_KANA_DICT_FILE", "tts_kana_dict.csv")
_tts_kana_dict_cache: dict[str, str] | None = None

# 発声効果の絵文字の挙動分類（グローバル設定＝キャラ非依存）。コード既定を土台に、
# tts_emoji/emoji_behavior.json があれば絵文字単位で上書きする。
TTS_EMOJI_ROOT = APP_ROOT / "tts_emoji"
TTS_EMOJI_BEHAVIOR_FILE = os.environ.get("TTS_EMOJI_BEHAVIOR_FILE", "emoji_behavior.json")
_emoji_behavior_cache: dict | None = None

# 感情セグメントを 1 発話でまとめて生成する際の、1 チャンクあたり最大文字数。
# 「はい。」のような極端に短い単独チャンクは TTS が末尾で暴走（言い直し・意味不明音）
# しやすいため、セグメント内の文をこの文字数以内で連結して 1 発話にする。超過する長い
# セグメントだけ複数チャンクへ分割する。Irodori の max_text_len(=256 token) 超過による
# テキスト切り捨てを避ける安全上限も兼ねるので、これより大幅に大きくしないこと。
TTS_SEGMENT_MAX_CHARS = int(os.environ.get("TTS_SEGMENT_MAX_CHARS", "120"))

IRODORI_CHECKPOINT = os.environ.get(
    "IRODORI_CHECKPOINT", "Aratako/Irodori-TTS-600M-v3-VoiceDesign"
)
DEFAULT_IRODORI_RUNTIME = "auto"
IRODORI_MODEL_DEVICE = os.environ.get(
    "IRODORI_MODEL_DEVICE",
    os.environ.get("IRODORI_DEVICE", DEFAULT_IRODORI_RUNTIME),
).strip() or DEFAULT_IRODORI_RUNTIME
IRODORI_MODEL_PRECISION = os.environ.get(
    "IRODORI_MODEL_PRECISION",
    os.environ.get("IRODORI_PRECISION", DEFAULT_IRODORI_RUNTIME),
).strip() or DEFAULT_IRODORI_RUNTIME
IRODORI_CODEC_DEVICE = os.environ.get(
    "IRODORI_CODEC_DEVICE",
    os.environ.get("IRODORI_DEVICE", DEFAULT_IRODORI_RUNTIME),
).strip() or DEFAULT_IRODORI_RUNTIME
IRODORI_CODEC_PRECISION = os.environ.get(
    "IRODORI_CODEC_PRECISION",
    os.environ.get("IRODORI_PRECISION", DEFAULT_IRODORI_RUNTIME),
).strip() or DEFAULT_IRODORI_RUNTIME
IRODORI_CAPTION = os.environ.get(
    "IRODORI_CAPTION",
    (
        "Native Japanese young adult woman, cute anime assistant voice, "
        "warm and intimate conversational acting, slightly teasing little-devil smile, "
        "soft breath, gentle emotional nuance, clear pronunciation, clean studio sound."
    ),
)
# CFG Scale の初期値。キャラクターごとに上書き可能で、未指定時はこの値にフォールバックする。
# 感情セグメントごとに caption が変わる構成では、未設定キャラの音色がリファレンス話者から
# 離れ「一部の文だけ別人の声」になりやすい。そのため speaker の既定値を高めにして音色を
# リファレンスへ寄せる。caption は感情表現に効くため据え置き（キャラ別に上書き可）。
IRODORI_CFG_SCALE_TEXT = float(os.environ.get("IRODORI_CFG_SCALE_TEXT", "3.0"))
IRODORI_CFG_SCALE_CAPTION = float(os.environ.get("IRODORI_CFG_SCALE_CAPTION", "4.0"))
IRODORI_CFG_SCALE_SPEAKER = float(os.environ.get("IRODORI_CFG_SCALE_SPEAKER", "6.5"))
IRODORI_REF_WAV = Path(
    os.environ.get(
        "IRODORI_REF_WAV",
        str(APP_ROOT / "static" / "reference" / "tokyo_ref.wav"),
    )
)
LUVIA_REF_WAV = Path(
    os.environ.get(
        "LUVIA_REF_WAV",
        str(APP_ROOT / "static" / "reference" / "luvia_smoky_radio_pitchdown3_ref.wav"),
    )
)
LUVIA_REMOTE_TTS_HOST = os.environ.get("LUVIA_REMOTE_TTS_HOST", "").strip()
LUVIA_REMOTE_IRODORI_ROOT = os.environ.get("LUVIA_REMOTE_IRODORI_ROOT", "").strip()
LUVIA_REMOTE_REF_WAV = os.environ.get(
    "LUVIA_REMOTE_REF_WAV",
    "",
)
LUVIA_REMOTE_TTS_URL = os.environ.get("LUVIA_REMOTE_TTS_URL", "").strip().rstrip("/")
LUVIA_REMOTE_DEFAULT_PORT = int(os.environ.get("LUVIA_REMOTE_DEFAULT_PORT", "7874"))
ALLOWED_REFERENCE_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

Irodori_lock = threading.Lock()
Irodori_module = None
Emoji_items_cache = None
Luvia_remote_ref_cache: dict[str, str] = {}
External_speak_lock = threading.Lock()
External_speak_events = deque(maxlen=80)
External_speak_next_id = 0
Codex_inbox_lock = threading.Lock()
# chat.jsonl / chat_emotion.jsonl の追記と、削除による書き換えを排他する。
# 削除は読み込み→絞り込み→全書き戻しなので、その途中に追記が入ると行を失う。
Chat_log_lock = threading.Lock()
Codex_inbox = deque(maxlen=120)
Codex_inbox_next_id = 0

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.nn\.utils\.weight_norm` is deprecated in favor of `torch\.nn\.utils\.parametrizations\.weight_norm`.*",
    category=FutureWarning,
)


def suppress_irodori_log_line(line: str) -> bool:
    return line.startswith(
        (
            "Using the default SDR of ",
            "WARNING! Reducing the sampling rate of the original audio from ",
        )
    )


class FilteredIrodoriStdout:
    def __init__(self, target) -> None:
        self.target = target
        self.buffer = ""

    def write(self, text: str) -> int:
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if not suppress_irodori_log_line(line):
                self.target.write(f"{line}\n")
        return len(text)

    def flush(self) -> None:
        if self.buffer:
            if not suppress_irodori_log_line(self.buffer):
                self.target.write(self.buffer)
            self.buffer = ""
        self.target.flush()


def json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def split_sentences(text: str, limit: int = 8) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？!?])\s*", text)
    chunks = [part.strip() for part in parts if part.strip()]
    if len(chunks) <= limit:
        return chunks
    return chunks[: limit - 1] + ["".join(chunks[limit - 1 :])]


def group_sentences(text: str, max_chars: int, limit: int = 20) -> list[str]:
    """文を「1 グループ ≦ max_chars 文字」になるよう貪欲に連結する。

    感情セグメントを 1 発話でまとめて生成するための分割。短いセグメントは 1 グループ
    （＝1 発話）にまとまり、TTS が短文で起こす末尾の暴走（言い直し・意味不明音）を避ける。
    1 文だけで max_chars を超える場合はその文を単独グループにする。グループ数が limit を
    超えたら末尾をまとめて limit 個に収める。空文字なら空リストを返す。
    """
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？!?])\s*", text)
    sentences = [part.strip() for part in parts if part.strip()]
    if not sentences:
        return []
    cap = max(1, int(max_chars))
    groups: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + len(sentence) <= cap:
            current += sentence
        else:
            groups.append(current)
            current = sentence
    if current:
        groups.append(current)
    if len(groups) > limit:
        groups = groups[: limit - 1] + ["".join(groups[limit - 1 :])]
    return groups


def load_emoji_items() -> list[dict[str, str]]:
    global Emoji_items_cache
    if Emoji_items_cache is not None:
        return Emoji_items_cache
    sys.path.insert(0, str(IRODORI_ROOT))
    from irodori_tts.gradio_emoji_palette import EMOJI_PALETTE_ITEMS

    Emoji_items_cache = [
        {
            "emoji": item.emoji,
            "label": item.label,
            "description": item.description,
        }
        for item in EMOJI_PALETTE_ITEMS
    ]
    return Emoji_items_cache


def safe_load_emoji_items() -> list[dict[str, str]]:
    try:
        return load_emoji_items()
    except Exception:
        return []


# コード既定はパレットの「ラベル」で分類を持つ（絵文字グリフの表記ゆれ＝異体字セレクタ等に
# 影響されないため）。マップのキーは実行時にパレットの絵文字へ解決するので、LLM が返す
# seg_emoji（パレット検証済み）と必ず同一表現で突き合わせできる。
#   singleShot = 非言語の単発音（先頭チャンクのみに付与し、文ごとの繰り返し挿入を防ぐ）
#   sustained  = 話し方・声色・音響効果（全チャンクに付与し、セグメント途中の表現ぶれを防ぐ）
_EMOJI_SINGLE_SHOT_LABELS: frozenset[str] = frozenset({
    "吐息", "間", "笑い", "喘ぎ", "息をのむ", "舐める音", "リップノイズ", "泣き声",
    "悲鳴", "寝言", "飲み込む", "咳・鼻", "舌打ち", "驚き", "あくび", "相槌", "鼻歌", "嗅ぐ音",
})
_EMOJI_SUSTAINED_LABELS: frozenset[str] = frozenset({
    "囁き", "エコー", "からかう", "震え声", "息切れ", "優しく", "眠そう", "早口", "電話越し",
    "ゆっくり", "慌てる", "喜び", "勢いよく", "怒り", "苦しげ", "心配", "照れ", "呆れ",
    "楽しげ", "得意げ", "懇願", "酔う", "口を塞ぐ", "安堵", "疑問", "力強く", "朗読",
})
_EMOJI_DEFAULT_FOR_UNKNOWN = "singleShot"


def _emoji_log(msg: str) -> None:
    """絵文字を含むログを、cp932 等の非 UTF-8 コンソールでも落ちないよう安全に出力する。

    通常は絵文字をそのまま出す（UTF-8 コンソールで可読）。エンコードできない環境では
    バックスラッシュエスケープへフォールバックし、UnicodeEncodeError で処理を止めない。
    """
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(msg.encode(enc, "backslashreplace").decode(enc, "replace"), flush=True)


def _resolve_emoji_behavior_file() -> Path:
    """絵文字挙動ファイルのパスを解決する（tts_emoji/ → プロジェクト直下 → cwd の順）。"""
    name = str(TTS_EMOJI_BEHAVIOR_FILE)
    path = Path(name)
    if path.is_absolute():
        return path
    return next(
        (base / name for base in (TTS_EMOJI_ROOT, APP_ROOT, Path.cwd()) if (base / name).exists()),
        TTS_EMOJI_ROOT / name,
    )


def _load_emoji_behavior() -> dict:
    """絵文字→挙動（singleShot/sustained）の対応と未分類既定を読み込む（キャッシュ）。

    コード既定（パレットのラベル分類）を土台に、``tts_emoji/emoji_behavior.json`` があれば
    絵文字単位で上書き（マージ）する。両リストに重複した絵文字は singleShot を優先。
    パレットに無い絵文字は無視してログする。未分類（新絵文字を含む）は defaultForUnknown。
    """
    global _emoji_behavior_cache
    if _emoji_behavior_cache is not None:
        return _emoji_behavior_cache

    palette_items = safe_load_emoji_items()
    palette = {item["emoji"] for item in palette_items if item.get("emoji")}

    mapping: dict[str, str] = {}
    unclassified: list[str] = []
    for item in palette_items:
        emoji = str(item.get("emoji") or "")
        label = str(item.get("label") or "")
        if not emoji:
            continue
        if label in _EMOJI_SUSTAINED_LABELS:
            mapping[emoji] = "sustained"
        elif label in _EMOJI_SINGLE_SHOT_LABELS:
            mapping[emoji] = "singleShot"
        else:
            mapping[emoji] = _EMOJI_DEFAULT_FOR_UNKNOWN
            unclassified.append(f"{emoji}({label})")

    default_unknown = _EMOJI_DEFAULT_FOR_UNKNOWN
    path = _resolve_emoji_behavior_file()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))

            def _apply(key: str, behavior: str) -> None:
                for emoji in data.get(key) or []:
                    emoji = str(emoji).strip()
                    if not emoji:
                        continue
                    if palette and emoji not in palette:
                        _emoji_log(f"[emoji-behavior] '{emoji}' not in palette; ignored")
                        continue
                    mapping[emoji] = behavior

            # sustained を先、singleShot を後に適用 → 重複時は singleShot が優先される。
            _apply("sustained", "sustained")
            _apply("singleShot", "singleShot")
            du = str(data.get("defaultForUnknown") or "").strip()
            if du in ("singleShot", "sustained"):
                default_unknown = du
            _emoji_log(f"[emoji-behavior] loaded overrides from {path}")
        except Exception as exc:
            _emoji_log(f"[emoji-behavior] load failed ({path}): {exc}")

    if unclassified:
        _emoji_log(
            f"[emoji-behavior] unclassified (default={default_unknown}): {' '.join(unclassified)}"
        )

    result = {"map": mapping, "defaultForUnknown": default_unknown}
    # パレット未取得（Irodori 未ロード等）ならキャッシュせず、次回に再構築させる。
    if palette:
        _emoji_behavior_cache = result
    return result


def emoji_is_sustained(emoji: str) -> bool:
    """絵文字が「持続系（全チャンクに付与すべき話し方/声色/音響効果）」なら True。

    単発の効果音（既定）と空文字は False。判定不能な未分類は defaultForUnknown に従う。
    """
    key = str(emoji or "").strip()
    if not key:
        return False
    behavior = _load_emoji_behavior()
    return behavior["map"].get(key, behavior["defaultForUnknown"]) == "sustained"


def enqueue_codex_inbox(payload: dict) -> dict:
    global Codex_inbox_next_id
    with Codex_inbox_lock:
        Codex_inbox_next_id += 1
        item = {
            "id": Codex_inbox_next_id,
            "createdAt": time.time(),
            **payload,
        }
        Codex_inbox.append(item)
    return item


def codex_inbox_since(after: int) -> list[dict]:
    with Codex_inbox_lock:
        return [item for item in Codex_inbox if int(item.get("id", 0)) > after]


def apply_emoji_style(text: str, emoji_style: str) -> str:
    text = strip_irodori_style_marks(text)
    emoji_style = str(emoji_style or "").strip()
    if not emoji_style:
        return text
    return f"{emoji_style}{text}"


# --- 英字→カナ変換（TTS 直前でのみ使用。保存・表示テキストは変換しない）------------------
# 未知語フォールバック用のローマ字系カナ表（英語読みの近似。正確さより「暴走させない」が目的）。
def _build_kana_syllable_table() -> dict[str, str]:
    rows = {
        "": "アイウエオ",
        "k": "カキクケコ", "g": "ガギグゲゴ", "s": "サシスセソ", "z": "ザジズゼゾ",
        "t": "タチツテト", "d": "ダヂヅデド", "n": "ナニヌネノ", "h": "ハヒフヘホ",
        "b": "バビブベボ", "p": "パピプペポ", "m": "マミムメモ", "r": "ラリルレロ",
        "y": "ヤ・ユ・ヨ", "w": "ワ・・・ヲ",
    }
    table: dict[str, str] = {}
    for consonant, kana_row in rows.items():
        for vowel, kana in zip("aiueo", kana_row):
            if kana != "・":
                table[consonant + vowel] = kana
    # 英単語向けの追加読み（デジラフ・外来音）。
    table.update({
        "shi": "シ", "sha": "シャ", "shu": "シュ", "sho": "ショ", "she": "シェ",
        "chi": "チ", "cha": "チャ", "chu": "チュ", "cho": "チョ", "che": "チェ",
        "tsu": "ツ", "si": "シ", "ti": "ティ", "tu": "トゥ", "di": "ディ", "du": "ドゥ",
        "fa": "ファ", "fi": "フィ", "fu": "フ", "fe": "フェ", "fo": "フォ",
        "ja": "ジャ", "ji": "ジ", "ju": "ジュ", "je": "ジェ", "jo": "ジョ",
        "va": "ヴァ", "vi": "ヴィ", "vu": "ヴ", "ve": "ヴェ", "vo": "ヴォ",
        "la": "ラ", "li": "リ", "lu": "ル", "le": "レ", "lo": "ロ",
        "wi": "ウィ", "we": "ウェ", "wo": "ウォ", "wu": "ウ", "ye": "イェ",
        "the": "ザ", "tha": "サ", "thi": "シ", "tho": "ソ",
    })
    return table


_KANA_SYLLABLES = _build_kana_syllable_table()
# 子音単独（子音クラスタ/語末）→ 母音を補ったカナ。
_KANA_CONSONANT_ONLY = {
    "k": "ク", "g": "グ", "s": "ス", "z": "ズ", "t": "ト", "d": "ド", "n": "ン",
    "h": "フ", "b": "ブ", "p": "プ", "m": "ム", "r": "ル", "l": "ル", "y": "イ",
    "w": "ウ", "c": "ク", "f": "フ", "j": "ジュ", "v": "ヴ", "x": "クス", "q": "ク",
}
# 最終ガード用: 変換後に残ったラテン文字を 1 字ずつアルファベット読みのカナへ。
_KANA_LETTER_NAMES = {
    "a": "エー", "b": "ビー", "c": "シー", "d": "ディー", "e": "イー", "f": "エフ",
    "g": "ジー", "h": "エイチ", "i": "アイ", "j": "ジェー", "k": "ケー", "l": "エル",
    "m": "エム", "n": "エヌ", "o": "オー", "p": "ピー", "q": "キュー", "r": "アール",
    "s": "エス", "t": "ティー", "u": "ユー", "v": "ブイ", "w": "ダブリュー",
    "x": "エックス", "y": "ワイ", "z": "ゼット",
}
_VOWELS = frozenset("aiueo")


def _romaji_to_kana(word: str) -> str:
    """英字の並びをカナへ近似変換する（辞書に無い未知語のフォールバック）。"""
    s = word.lower()
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        matched = False
        for length in (3, 2):  # 長い綴り(shi, cha, the...) を優先
            if s[i:i + length] in _KANA_SYLLABLES:
                out.append(_KANA_SYLLABLES[s[i:i + length]])
                i += length
                matched = True
                break
        if matched:
            continue
        ch = s[i]
        if ch in _VOWELS:
            out.append(_KANA_SYLLABLES[ch])
        elif ch in _KANA_CONSONANT_ONLY:
            out.append(_KANA_CONSONANT_ONLY[ch])
        else:
            out.append(_KANA_LETTER_NAMES.get(ch, ""))
        i += 1
    return "".join(out)


def _load_one_kana_dict_file(path: Path, mapping: dict[str, str]) -> None:
    """1 つの辞書ファイルを ``mapping`` へ読み込む（同一キーは後勝ち）。"""
    if not path.exists():
        print(f"[tts-kana] dict not found ({path}); skipped", flush=True)
        return
    count = 0
    # utf-8-sig で読むと、Excel 等が付ける BOM を自動除去する（BOM なしファイルも同じ結果）。
    # これにより「BOM付き UTF-8」で保存された辞書でも 1 行目が壊れず読み込める。
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[,\t]", line, maxsplit=1)
        if len(parts) != 2:
            continue
        key, value = parts[0].strip().lower(), parts[1].strip()
        if key and value:
            mapping[key] = value
            count += 1
    print(f"[tts-kana] loaded {count} entries from {path}", flush=True)


def _load_tts_kana_dict() -> dict[str, str]:
    """英字→カナ辞書をファイルから読み込む（キャッシュ）。

    ``TTS_KANA_DICT_FILE`` は ``;`` 区切りで複数指定でき、**後に書いたファイルの内容が優先**
    される（例 ``alkanadict.csv;tts_kana_dict.csv`` なら alkana を土台に、独自用語辞書で上書き）。
    各ファイルは 1 行 ``english,カタカナ``（カンマ or タブ区切り、``#`` はコメント）。
    alkana の外部データ CSV と同一形式。ファイルが無くても空辞書として続行する。
    """
    global _tts_kana_dict_cache
    if _tts_kana_dict_cache is not None:
        return _tts_kana_dict_cache
    mapping: dict[str, str] = {}
    names = [n.strip() for n in str(TTS_KANA_DICT_FILE).split(";") if n.strip()]
    for name in names:
        path = Path(name)
        if not path.is_absolute():
            # 「alkanadict.csv」(tts_dictionaries 配下) も
            # 「tts_dictionaries/alkanadict.csv」(プロジェクト相対) も受け付ける。
            path = next(
                (base / name for base in (TTS_DICT_ROOT, APP_ROOT, Path.cwd())
                 if (base / name).exists()),
                TTS_DICT_ROOT / name,
            )
        try:
            _load_one_kana_dict_file(path, mapping)
        except Exception as exc:  # 辞書が壊れていても TTS は止めない
            print(f"[tts-kana] dict load failed ({path}): {exc}", flush=True)
    print(f"[tts-kana] total {len(mapping)} entries", flush=True)
    _tts_kana_dict_cache = mapping
    return mapping


def english_to_kana_for_tts(text: str) -> str:
    """TTS へ渡す直前に、テキスト中のラテン文字だけをカナへ置き換える。

    保存・表示テキストには使わないこと（原文を残す）。辞書→未知語フォールバックの順で
    変換し、最後に残ったラテン文字を必ずカナへ落として Irodori に英字を渡さない。
    """
    if TTS_KANA_NORMALIZE == "off" or not text or not re.search(r"[A-Za-z]", text):
        return text
    dictionary = _load_tts_kana_dict()

    def _replace_word(match: "re.Match[str]") -> str:
        word = match.group(0)
        return dictionary.get(word.lower()) or _romaji_to_kana(word)

    converted = re.sub(r"[A-Za-z]+", _replace_word, text)
    # 最終ガード: 何らかの理由で残ったラテン文字を 1 字ずつカナへ（暴走の構造的防止）。
    return re.sub(
        r"[A-Za-z]",
        lambda m: _KANA_LETTER_NAMES.get(m.group(0).lower(), ""),
        converted,
    )


def expression_for_emoji(emoji_style: str) -> str:
    emoji = str(emoji_style or "").strip()
    if not emoji:
        return "neutral"
    mapping = {
        "👂": "soft",
        "😮‍💨": "sigh",
        "⏸️": "pause",
        "🤭": "teasing",
        "🥵": "breathless",
        "📢": "broadcast",
        "😏": "teasing",
        "🥺": "worried",
        "🌬️": "breathless",
        "😮": "gasp",
        "👅": "muffled",
        "💋": "muffled",
        "🫶": "tender",
        "😭": "sad",
        "😱": "surprised",
        "😪": "sleepy",
        "😴": "sleepy",
        "⏩": "fast",
        "📞": "phone",
        "🐢": "sleepy",
        "🥤": "swallow",
        "🤧": "cough",
        "😒": "exasperated",
        "😰": "worried",
        "😆": "happy",
        "💥": "strong",
        "😠": "angry",
        "😲": "gasp",
        "🥱": "yawn",
        "😖": "worried",
        "😟": "worried",
        "🫣": "shy",
        "🙄": "exasperated",
        "😊": "happy",
        "😎": "smug",
        "👌": "neutral",
        "🙏": "pleading",
        "🥴": "muffled",
        "🎵": "humming",
        "🤐": "pause",
        "😌": "tender",
        "🤔": "question",
        "💪": "strong",
        "👃": "sniff",
        "📖": "narration",
    }
    return mapping.get(emoji, "neutral")


def expression_assets() -> dict[str, str | list[str]]:
    names = [
        "neutral",
        "happy",
        "surprised",
        "soft",
        "angry",
        "worried",
        "sad",
        "shy",
        "narration",
        "fast",
        "sleepy",
        "phone",
        "echo",
        "muffled",
        "throat",
        "strong",
        "teasing",
        "pleading",
        "exasperated",
        "smug",
        "sigh",
        "gasp",
        "breathless",
        "yawn",
        "humming",
        "swallow",
        "cough",
        "sniff",
        "pause",
        "question",
        "tender",
        "broadcast",
    ]
    assets: dict[str, str | list[str]] = {}
    expression_dir = STATIC_ROOT / "expressions"
    for name in names:
        variants = sorted(expression_dir.glob(f"{name}*.png"))
        urls = [f"/expressions/{path.name}" for path in variants if not path.name.endswith("_sheet.png")]
        assets[name] = urls if len(urls) > 1 else (urls[0] if urls else f"/expressions/{name}.png")
    return assets


def expression_asset_lists(root: Path, url_prefix: str) -> dict[str, list[str]]:
    assets: dict[str, list[str]] = {}
    if not root.exists():
        return assets
    for path in sorted(root.glob("*.png")):
        if path.name.endswith("_sheet.png") or path.name.endswith("_contact.png"):
            continue
        key = re.sub(r"_\d+$", "", path.stem)
        if key.startswith("luvia_"):
            key = key.removeprefix("luvia_")
        assets.setdefault(key, []).append(f"{url_prefix}/{path.name}")
    return assets


def default_character_profiles() -> dict:
    rinon_expressions = expression_asset_lists(STATIC_ROOT / "expressions", "/expressions")
    luvia_expressions = expression_asset_lists(
        STATIC_ROOT / "second_player" / "expressions",
        "/second_player/expressions",
    )
    return {
        "version": 1,
        "activeMainId": "rinon",
        "activeSecondId": "luvia",
        "characters": [
            {
                "id": "rinon",
                "name": "リノン",
                "systemPrompt": (
                    "リノンは20歳以上の、アニメ的で少し色っぽい日本語の女の子AI。"
                    "人なつっこく、気が利き、相手の反応を見ながら甘くからかったり照れたりする。"
                    "会話は自然で短めに返し、距離感は近いが、露骨すぎる性的描写や未成年っぽい振る舞いは避ける。"
                    "声は明るくやわらかく、少し小悪魔っぽい余裕と、ふとした照れを混ぜる。"
                ),
                "ttsCaption": IRODORI_CAPTION,
                "styleGuide": "",
                "steps": DEFAULT_CHARACTER_STEPS,
                "cfgScaleText": IRODORI_CFG_SCALE_TEXT,
                "cfgScaleCaption": IRODORI_CFG_SCALE_CAPTION,
                "cfgScaleSpeaker": IRODORI_CFG_SCALE_SPEAKER,
                "referencePath": str(IRODORI_REF_WAV),
                "portrait": "/expressions/neutral.png",
                "expressions": rinon_expressions,
            },
            {
                "id": "luvia",
                "name": "ルヴィア",
                "systemPrompt": (
                    "ルヴィアは20歳以上の、赤髪で勝ち気なアニメ的美少女AI。"
                    "リノンより少しストレートで、挑発的だが根は面倒見がよい。"
                    "会話では相手をからかいながらも、要点ははっきり伝える。"
                    "露骨すぎる性的描写や未成年っぽい振る舞いは避ける。"
                    "声は少し低めで明るく、元気で自信があり、いたずらっぽい笑みを含む。"
                ),
                "ttsCaption": (
                    "Native Japanese adult woman, smoky radio presenter voice, low feminine resonance, "
                    "confident lively tone, polished adult speaking style, restrained teasing confidence, "
                    "clean studio sound."
                ),
                "styleGuide": "",
                "steps": DEFAULT_CHARACTER_STEPS,
                "cfgScaleText": IRODORI_CFG_SCALE_TEXT,
                "cfgScaleCaption": IRODORI_CFG_SCALE_CAPTION,
                "cfgScaleSpeaker": IRODORI_CFG_SCALE_SPEAKER,
                "referencePath": str(LUVIA_REF_WAV),
                "portrait": "/second_player/expressions/luvia_neutral.png",
                "expressions": luvia_expressions,
            },
        ],
    }


def sanitize_cfg_scale(value: object, default: float) -> float:
    """CFG Scale 値を float に正規化し、0〜20 の範囲へクランプする。

    数値化できない／NaN の場合は ``default`` を返す。
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if result != result:  # NaN
        return float(default)
    return max(0.0, min(20.0, result))


# キャラ別 Num Steps の既定。UI のグローバル steps 欄より、キャラ設定があればそちらを優先する。
# 低ステップ（従来の 12）だとリファレンス話者の条件付けが解像しきれず音色が寄り切らないため、
# キャラ別既定は 40 とする。
DEFAULT_CHARACTER_STEPS = 40


def sanitize_steps(value: object, default: int = DEFAULT_CHARACTER_STEPS) -> int:
    """Num Steps を int に正規化し、1〜120 の範囲へクランプする。

    数値化できない場合は ``default`` を返す。
    """
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        return int(default)
    return max(1, min(120, result))


def sanitize_character_id(value: object, fallback: str = "") -> str:
    raw = str(value or "").strip().lower()
    safe = re.sub(r"[^0-9a-z_-]+", "_", raw).strip("_")
    return safe[:48] or fallback or f"character_{uuid.uuid4().hex[:8]}"


def sanitize_expression_key(value: object, fallback: str = "neutral") -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", str(value or fallback)).strip("_") or fallback


def character_url(character_id: str, *parts: str) -> str:
    return "/".join(["/Character", character_id, *[part.strip("/\\") for part in parts if part]])


def local_path_for_asset_url(url: str) -> Path | None:
    text = str(url or "").strip()
    if not text.startswith("/"):
        return None
    if text.startswith("/Character/"):
        rel = Path(unquote(text.removeprefix("/Character/")))
        return (CHARACTER_ROOT / rel).resolve()
    if text.startswith("/characters/"):
        rel = Path(unquote(text.removeprefix("/characters/")))
        return (LEGACY_CHARACTER_ROOT / rel).resolve()
    rel = Path(unquote(text.lstrip("/")))
    return (STATIC_ROOT / rel).resolve()


def copy_character_asset(character_id: str, expression: str, url: str) -> str:
    text = str(url or "").strip()
    if text.startswith(f"/Character/{character_id}/"):
        return text
    src = local_path_for_asset_url(text)
    if not src or not src.exists() or src.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        return text
    expression_key = sanitize_expression_key(expression)
    out_dir = CHARACTER_ROOT / character_id / "expressions" / expression_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / src.name
    if out_path.exists() and out_path.resolve() != src.resolve():
        if out_path.read_bytes() != src.read_bytes():
            out_path = out_dir / f"{src.stem}_{uuid.uuid4().hex[:6]}{src.suffix.lower()}"
    if not out_path.exists():
        shutil.copy2(src, out_path)
    return character_url(character_id, "expressions", expression_key, out_path.name)


def copy_character_reference(character_id: str, reference_path: str) -> str:
    raw = str(reference_path or "").strip()
    if not raw:
        return raw
    src = Path(raw)
    char_dir = (CHARACTER_ROOT / character_id).resolve()
    if not src.is_absolute():
        char_relative = (char_dir / src).resolve()
        src = char_relative if char_relative.exists() else (APP_ROOT / src).resolve()
    if not src.exists() or src.suffix.lower() not in ALLOWED_REFERENCE_EXTENSIONS:
        return raw
    if str(src.resolve()).startswith(str(char_dir)):
        return str(src.resolve())
    out_dir = CHARACTER_ROOT / character_id / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / src.name
    if out_path.exists() and out_path.resolve() != src.resolve():
        if out_path.read_bytes() != src.read_bytes():
            out_path = out_dir / f"{src.stem}_{uuid.uuid4().hex[:6]}{src.suffix.lower()}"
    if not out_path.exists():
        shutil.copy2(src, out_path)
    return str(out_path.resolve())


def character_reference_for_disk(character: dict) -> dict:
    disk_character = json.loads(json.dumps(character, ensure_ascii=False))
    character_id = sanitize_character_id(disk_character.get("id"))
    char_dir = (CHARACTER_ROOT / character_id).resolve()
    raw = str(disk_character.get("referencePath") or "").strip()
    if raw:
        try:
            ref_path = Path(raw).resolve()
            if str(ref_path).startswith(str(char_dir)):
                disk_character["referencePath"] = str(ref_path.relative_to(char_dir))
        except Exception:
            pass
    return disk_character


def character_text_profile(character: dict) -> str:
    expressions = character.get("expressions") if isinstance(character.get("expressions"), dict) else {}
    lines = [
        "# Rinon Voice Lab character profile",
        "# Edit this file, then use Options > キャラ読込 to reload.",
        f"id: {character.get('id', '')}",
        f"name: {character.get('name', '')}",
        f"referencePath: {character.get('referencePath', '')}",
        f"portrait: {character.get('portrait', '')}",
        f"cfgScaleText: {character.get('cfgScaleText', IRODORI_CFG_SCALE_TEXT)}",
        f"cfgScaleCaption: {character.get('cfgScaleCaption', IRODORI_CFG_SCALE_CAPTION)}",
        f"cfgScaleSpeaker: {character.get('cfgScaleSpeaker', IRODORI_CFG_SCALE_SPEAKER)}",
        f"steps: {character.get('steps', DEFAULT_CHARACTER_STEPS)}",
        "",
        "[systemPrompt]",
        str(character.get("systemPrompt") or ""),
        "",
        "[ttsCaption]",
        str(character.get("ttsCaption") or ""),
        "",
        "[styleGuide]",
        str(character.get("styleGuide") or ""),
        "",
        "[expressions]",
    ]
    for key in sorted(expressions):
        values = expressions.get(key) or []
        raw_values = values if isinstance(values, list) else [values]
        lines.append(f"{key}=" + "|".join(str(value) for value in raw_values if str(value or "").strip()))
    lines.append("")
    return "\n".join(lines)


def parse_character_text_profile(path: Path) -> dict:
    values: dict[str, str] = {}
    sections: dict[str, list[str]] = {"systemPrompt": [], "ttsCaption": [], "styleGuide": [], "expressions": []}
    section = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            sections.setdefault(section, [])
            continue
        if section in {"systemPrompt", "ttsCaption", "styleGuide"}:
            sections[section].append(line)
            continue
        if section == "expressions":
            sections[section].append(line)
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    expressions: dict[str, list[str]] = {}
    for line in sections.get("expressions", []):
        if "=" not in line:
            continue
        key, raw_values = line.split("=", 1)
        expr_key = sanitize_expression_key(key)
        expressions[expr_key] = [value.strip() for value in raw_values.split("|") if value.strip()]
    return {
        "id": values.get("id", path.parent.name),
        "name": values.get("name", path.parent.name),
        "referencePath": values.get("referencePath", ""),
        "portrait": values.get("portrait", ""),
        "cfgScaleText": sanitize_cfg_scale(values.get("cfgScaleText"), IRODORI_CFG_SCALE_TEXT),
        "cfgScaleCaption": sanitize_cfg_scale(values.get("cfgScaleCaption"), IRODORI_CFG_SCALE_CAPTION),
        "cfgScaleSpeaker": sanitize_cfg_scale(values.get("cfgScaleSpeaker"), IRODORI_CFG_SCALE_SPEAKER),
        "steps": sanitize_steps(values.get("steps"), DEFAULT_CHARACTER_STEPS),
        "systemPrompt": "\n".join(sections.get("systemPrompt", [])).strip(),
        "ttsCaption": "\n".join(sections.get("ttsCaption", [])).strip(),
        "styleGuide": "\n".join(sections.get("styleGuide", [])).strip(),
        "expressions": expressions,
    }


def save_character_folder_profile(character: dict) -> None:
    character_id = sanitize_character_id(character.get("id"))
    char_dir = CHARACTER_ROOT / character_id
    char_dir.mkdir(parents=True, exist_ok=True)
    disk_character = character_reference_for_disk(character)
    (char_dir / "profile.json").write_text(
        json.dumps(disk_character, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (char_dir / "profile.txt").write_text(character_text_profile(disk_character), encoding="utf-8")


def load_character_folder_profiles() -> list[dict]:
    profiles: list[dict] = []
    if not CHARACTER_ROOT.exists():
        return profiles
    for char_dir in sorted(path for path in CHARACTER_ROOT.iterdir() if path.is_dir()):
        text_path = char_dir / "profile.txt"
        json_path = char_dir / "profile.json"
        try:
            if text_path.exists() and (not json_path.exists() or text_path.stat().st_mtime >= json_path.stat().st_mtime):
                profiles.append(parse_character_text_profile(text_path))
            elif json_path.exists():
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    profiles.append(data)
        except Exception:
            continue
    return profiles


def materialize_character_profile(profile: dict) -> dict:
    clean = sanitize_character_profiles(profile, materialize=False)
    CHARACTER_ROOT.mkdir(parents=True, exist_ok=True)
    for character in clean["characters"]:
        character_id = character["id"]
        clean_expressions: dict[str, list[str]] = {}
        for key, values in (character.get("expressions") or {}).items():
            expr_key = sanitize_expression_key(key)
            clean_expressions[expr_key] = [copy_character_asset(character_id, expr_key, value) for value in values]
        character["expressions"] = clean_expressions
        portrait = str(character.get("portrait") or "").strip()
        if portrait:
            for values in clean_expressions.values():
                if portrait in values:
                    break
            else:
                portrait = copy_character_asset(character_id, "neutral", portrait)
        if not portrait:
            portrait = (clean_expressions.get("neutral") or ["/expressions/neutral.png"])[0]
        character["portrait"] = portrait
        character["referencePath"] = copy_character_reference(character_id, character.get("referencePath", ""))
        save_character_folder_profile(character)
    return clean


def sanitize_character_profiles(payload: dict, materialize: bool = True) -> dict:
    defaults = default_character_profiles()
    raw_characters = payload.get("characters") if isinstance(payload.get("characters"), list) else []
    characters: list[dict] = []
    used_ids: set[str] = set()
    for index, item in enumerate(raw_characters):
        if not isinstance(item, dict):
            continue
        char_id = sanitize_character_id(item.get("id"), f"character_{index + 1}")
        original_id = char_id
        suffix = 2
        while char_id in used_ids:
            char_id = f"{original_id}_{suffix}"
            suffix += 1
        used_ids.add(char_id)
        expressions = item.get("expressions") if isinstance(item.get("expressions"), dict) else {}
        clean_expressions: dict[str, list[str]] = {}
        for key, values in expressions.items():
            expr_key = sanitize_expression_key(key)
            raw_values = values if isinstance(values, list) else [values]
            urls = [str(url).strip() for url in raw_values if str(url or "").strip()]
            clean_expressions[expr_key] = urls[:80]
        portrait = str(item.get("portrait") or "").strip()
        if not portrait:
            neutral = clean_expressions.get("neutral") or []
            portrait = neutral[0] if neutral else "/expressions/neutral.png"
        characters.append(
            {
                "id": char_id,
                "name": str(item.get("name") or char_id).strip()[:80],
                "systemPrompt": str(item.get("systemPrompt") or "").strip(),
                "ttsCaption": str(item.get("ttsCaption") or IRODORI_CAPTION).strip(),
                "styleGuide": str(item.get("styleGuide") or "").strip(),
                "cfgScaleText": sanitize_cfg_scale(item.get("cfgScaleText"), IRODORI_CFG_SCALE_TEXT),
                "cfgScaleCaption": sanitize_cfg_scale(item.get("cfgScaleCaption"), IRODORI_CFG_SCALE_CAPTION),
                "cfgScaleSpeaker": sanitize_cfg_scale(item.get("cfgScaleSpeaker"), IRODORI_CFG_SCALE_SPEAKER),
                "steps": sanitize_steps(item.get("steps"), DEFAULT_CHARACTER_STEPS),
                "referencePath": str(item.get("referencePath") or IRODORI_REF_WAV).strip(),
                "portrait": portrait,
                "expressions": clean_expressions,
            }
        )
    if not characters:
        return defaults
    character_ids = {item["id"] for item in characters}
    active_main = sanitize_character_id(payload.get("activeMainId"), characters[0]["id"])
    active_second = sanitize_character_id(payload.get("activeSecondId"), characters[min(1, len(characters) - 1)]["id"])
    if active_main not in character_ids:
        active_main = characters[0]["id"]
    if active_second not in character_ids:
        active_second = characters[min(1, len(characters) - 1)]["id"]
    clean = {
        "version": 1,
        "activeMainId": active_main,
        "activeSecondId": active_second,
        "characters": characters,
    }
    return materialize_character_profile(clean) if materialize else clean


def load_character_profiles() -> dict:
    if CHARACTER_PROFILE_PATH.exists():
        data = json.loads(CHARACTER_PROFILE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = default_character_profiles()
    else:
        data = default_character_profiles()
    profile = sanitize_character_profiles(data, materialize=False)
    folder_profiles = load_character_folder_profiles()
    if folder_profiles:
        by_id = {character["id"]: character for character in profile["characters"]}
        order = [character["id"] for character in profile["characters"]]
        for raw_character in folder_profiles:
            sanitized = sanitize_character_profiles({"characters": [raw_character]}, materialize=False)["characters"][0]
            by_id[sanitized["id"]] = sanitized
            if sanitized["id"] not in order:
                order.append(sanitized["id"])
        profile["characters"] = [by_id[character_id] for character_id in order if character_id in by_id]
    profile = materialize_character_profile(profile)
    PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
    CHARACTER_PROFILE_PATH.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def save_character_profiles(payload: dict) -> dict:
    PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
    profile = sanitize_character_profiles(payload)
    CHARACTER_PROFILE_PATH.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return profile


def save_character_image(payload: dict) -> dict:
    character_id = sanitize_character_id(payload.get("characterId"))
    expression = sanitize_expression_key(payload.get("expression"))
    original_name = Path(str(payload.get("name") or "expression.png")).name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError("expression image must be png, jpg, jpeg, or webp")
    encoded = str(payload.get("dataBase64") or "")
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    image_bytes = base64.b64decode(encoded, validate=True)
    if not image_bytes:
        raise ValueError("expression image is empty")
    if len(image_bytes) > 32 * 1024 * 1024:
        raise ValueError("expression image is too large")
    out_dir = CHARACTER_ROOT / character_id / "expressions" / expression
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^0-9A-Za-z_-]+", "_", Path(original_name).stem).strip("_")[:48] or "image"
    file_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{safe_stem}{suffix}"
    out_path = out_dir / file_name
    out_path.write_bytes(image_bytes)
    return {
        "ok": True,
        "characterId": character_id,
        "expression": expression,
        "path": str(out_path),
        "url": character_url(character_id, "expressions", expression, file_name),
        "name": file_name,
        "size": out_path.stat().st_size,
    }


def append_chat_log(record: dict) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with Chat_log_lock:
        with CHAT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _match_deleted_turn(
    record: dict, *, user_text: str, reply_text: str, audio_url: str, speaker: str
) -> bool:
    """ログ 1 行が「削除された往復」かどうかを判定する。

    照合は確実な順に試す:
      1) 結合音声の URL（combinedUrl）— 往復ごとに一意なので最も確実
      2) ユーザー発言＋返答の一致
    返答は保存場所で形が違う（chat.jsonl は本文のみ、history.json は「話者名: 本文」）ため、
    どちらの形でも一致するように話者名を付けた形とも比べる。
    """
    if audio_url and str(record.get("combinedUrl") or "").strip() == audio_url:
        return True
    record_user = str(record.get("user") or "").strip()
    if not user_text or record_user != user_text:
        return False
    record_reply = str(record.get("reply") or "").strip()
    if not record_reply:
        return False
    record_speaker = str(record.get("speaker") or speaker or "").strip()
    candidates = {record_reply}
    if record_speaker:
        candidates.add(f"{record_speaker}: {record_reply}")
    return reply_text in candidates


def delete_turn_records(payload: dict) -> dict:
    """UI で削除された往復を、記憶系のファイルからまとめて取り除く。

    履歴（history.json）はクライアントの自動保存で更新されるが、それだけでは
      ・logs/chat.jsonl（一番大本の生ログ）
      ・logs/chat_emotion.jsonl（感情キャプション付き返答）
      ・profiles/sessions/<charId>/memory.sqlite3（RAG検索DB＋事実台帳）
    に残り続ける。特に chat.jsonl に残ると、後日そこから再構築したときに
    削除した会話が復活してしまうので、削除はこの 3 つまで揃えて初めて完了する。

    生成済みの音声ファイル（static/generated 配下）は消さない。他の履歴から参照されて
    いる可能性があり、消しても取り返せないため、掃除は別途手動で行う。
    """
    char_id = safe_character_id(payload.get("characterId"))
    user_text = str(payload.get("userText") or "").strip()
    reply_text = str(payload.get("replyText") or "").strip()
    audio_url = str(payload.get("audioUrl") or "").strip()
    speaker = str(payload.get("speaker") or "").strip()
    two_only = bool(payload.get("twoOnlyMode", False))
    if not user_text and not audio_url:
        return {"ok": False, "error": "userText か audioUrl が必要です"}

    result = {
        "ok": True,
        "characterId": char_id,
        "memories": 0,
        "facts": 0,
        "chatLog": 0,
        "emotionLog": 0,
    }

    # 1) RAG 検索DB（往復と、その往復から抽出した事実）
    if rag_memory is not None:
        try:
            mode = "two_only" if two_only else "normal"
            targets = [
                turn["id"]
                for turn in rag_memory.list_turns(char_id)
                if str(turn.get("user_text") or "").strip() == user_text
                and str(turn.get("reply_text") or "").strip() == reply_text
                and str(turn.get("mode") or "normal") == mode
            ]
            if targets:
                before = rag_memory.facts_stats(char_id)["count"]
                result["memories"] = rag_memory.delete_memories(char_id, targets)
                result["facts"] = max(
                    0, before - rag_memory.facts_stats(char_id)["count"]
                )
        except Exception as exc:
            result["ragError"] = f"{type(exc).__name__}: {exc}"

    # 2) chat.jsonl（大本の生ログ）— 書き換え前に .bak へ退避する
    with Chat_log_lock:
        if CHAT_LOG_PATH.exists():
            kept: list[str] = []
            removed = 0
            for line in CHAT_LOG_PATH.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except ValueError:
                    kept.append(line)  # 壊れた行は判断できないので温存する
                    continue
                if isinstance(record, dict) and _match_deleted_turn(
                    record,
                    user_text=user_text,
                    reply_text=reply_text,
                    audio_url=audio_url,
                    speaker=speaker,
                ):
                    removed += 1
                    continue
                kept.append(line)
            if removed:
                backup = CHAT_LOG_PATH.with_suffix(CHAT_LOG_PATH.suffix + ".bak")
                if not backup.exists():
                    shutil.copyfile(CHAT_LOG_PATH, backup)
                CHAT_LOG_PATH.write_text(
                    "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8"
                )
            result["chatLog"] = removed

        # 3) chat_emotion.jsonl（返答本文で紐づく）
        if EMOTION_LOG_PATH.exists():
            kept_emotion: list[str] = []
            removed_emotion = 0
            for line in EMOTION_LOG_PATH.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except ValueError:
                    kept_emotion.append(line)
                    continue
                if isinstance(record, dict) and _match_deleted_turn(
                    record,
                    user_text=user_text,
                    reply_text=reply_text,
                    audio_url="",  # 感情ログは combinedUrl を持たない
                    speaker=speaker,
                ):
                    removed_emotion += 1
                    continue
                kept_emotion.append(line)
            if removed_emotion:
                backup = EMOTION_LOG_PATH.with_suffix(EMOTION_LOG_PATH.suffix + ".bak")
                if not backup.exists():
                    shutil.copyfile(EMOTION_LOG_PATH, backup)
                EMOTION_LOG_PATH.write_text(
                    "\n".join(kept_emotion) + ("\n" if kept_emotion else ""),
                    encoding="utf-8",
                )
            result["emotionLog"] = removed_emotion

    print(
        f"[delete] {char_id} memories={result['memories']} facts={result['facts']} "
        f"chat.jsonl={result['chatLog']} emotion={result['emotionLog']} "
        f"user={compact_text(user_text, 24)!r}"
    )
    return result


def append_emotion_log(record: dict) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with Chat_log_lock:
        with EMOTION_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def chat_log_summary(limit: int = 20) -> dict:
    expression_counts: dict[str, int] = {}
    emoji_counts: dict[str, int] = {}
    recent: list[dict] = []
    total = 0
    if not CHAT_LOG_PATH.exists():
        return {
            "path": str(CHAT_LOG_PATH),
            "total": 0,
            "expressions": [],
            "emojis": [],
            "recent": [],
        }
    for line in CHAT_LOG_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        expression = str(item.get("expression") or "neutral")
        emoji = str(item.get("emojiStyle") or "")
        expression_counts[expression] = expression_counts.get(expression, 0) + 1
        if emoji:
            emoji_counts[emoji] = emoji_counts.get(emoji, 0) + 1
        recent.append(
            {
                "time": item.get("time"),
                "user": item.get("user"),
                "reply": item.get("reply"),
                "emojiStyle": emoji,
                "expression": expression,
                "chunks": item.get("chunkCount"),
            }
        )
        if len(recent) > limit:
            recent = recent[-limit:]
    return {
        "path": str(CHAT_LOG_PATH),
        "total": total,
        "expressions": sorted(
            ({"name": key, "count": value} for key, value in expression_counts.items()),
            key=lambda item: item["count"],
            reverse=True,
        ),
        "emojis": sorted(
            ({"emoji": key, "count": value} for key, value in emoji_counts.items()),
            key=lambda item: item["count"],
            reverse=True,
        ),
        "recent": recent,
    }


def sanitize_history(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    history: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        entry: dict = {"role": role, "content": content}
        # RAG 記憶の会話モードを表示外メタとして保持する（display と同じく LM 文脈では
        # 無視される）。手作業や再構築で付けた "mode"/"speaker" を保存往復で失わない
        # ため、正当な値だけ引き継ぐ。詳細は write_character_history のマージ処理参照。
        mode = str(item.get("mode") or "").strip()
        if mode in {"normal", "two_only"}:
            entry["mode"] = mode
        speaker = str(item.get("speaker") or "").strip()
        if speaker:
            entry["speaker"] = speaker
        ts = str(item.get("ts") or "").strip()
        if ts:
            entry["ts"] = ts
        # アシスタント返答は、リロード後も注釈（感情キャプション）・meta 行・再生対象を
        # 復元できるよう表示用メタを保持する。LM context（content）とは別物で、
        # /api/chat の文脈生成では無視される。
        display = item.get("display")
        if role == "assistant" and isinstance(display, dict):
            clean_display = {
                "text": str(display.get("text") or ""),
                "meta": str(display.get("meta") or ""),
                "audioUrl": str(display.get("audioUrl") or ""),
            }
            if any(clean_display.values()):
                entry["display"] = clean_display
        history.append(entry)
    return history


def sanitize_histories(value: object) -> dict[str, list[dict]]:
    """キャラクター ID をキーにした会話ログのマップを検証・整形する。"""
    result: dict[str, list[dict]] = {}
    if isinstance(value, dict):
        for key, entries in value.items():
            char_id = str(key or "").strip()
            if not char_id:
                continue
            result[char_id] = sanitize_history(entries)
    return result


def safe_character_id(value: object, fallback: str = "rinon") -> str:
    """キャラクター ID をフォルダ名に使える安全な文字列へ整形する。"""
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", str(value or "").strip()).strip("_")
    return cleaned[:64] or fallback


def sanitize_context_limits(value: object) -> dict[str, int]:
    """キャラクター ID をキーにした context 上限値のマップを検証・整形する。"""
    result: dict[str, int] = {}
    if isinstance(value, dict):
        for key, raw in value.items():
            char_id = str(key or "").strip()
            if not char_id:
                continue
            try:
                num = int(raw)
            except (TypeError, ValueError):
                continue
            if num > 0:
                result[char_id] = num
    return result


def normalize_session_settings(settings: dict) -> dict:
    settings = settings if isinstance(settings, dict) else {}
    return {
        "systemPrompt": str(settings.get("systemPrompt") or ""),
        "mainCharacterName": str(settings.get("mainCharacterName") or "リノン"),
        "secondCharacterName": str(settings.get("secondCharacterName") or "ルヴィア"),
        "activeMainCharacterId": str(settings.get("activeMainCharacterId") or "rinon"),
        "activeSecondCharacterId": str(settings.get("activeSecondCharacterId") or "luvia"),
        "userAddress": str(settings.get("userAddress") or "あなた"),
        "ttsCaption": str(settings.get("ttsCaption") or IRODORI_CAPTION),
        "secondSystemPrompt": str(settings.get("secondSystemPrompt") or ""),
        "secondTtsCaption": str(settings.get("secondTtsCaption") or IRODORI_CAPTION),
        "referencePath": str(settings.get("referencePath") or IRODORI_REF_WAV),
        "secondReferencePath": str(settings.get("secondReferencePath") or LUVIA_REF_WAV),
        "contextLimit": int(settings.get("contextLimit") or DEFAULT_CONTEXT_LIMIT),
        "characterContextLimits": sanitize_context_limits(settings.get("characterContextLimits")),
        "model": str(settings.get("model") or DEFAULT_MODEL),
        "steps": int(settings.get("steps") or 12),
        "speechRate": str(settings.get("speechRate") or "normal"),
        "replyLength": str(settings.get("replyLength") or "normal"),
        "llmGenerationMode": str(settings.get("llmGenerationMode") or DEFAULT_LM_GENERATION_MODE),
        "sendShortcut": str(settings.get("sendShortcut") or "enter"),
        "ttsBackendMode": str(settings.get("ttsBackendMode") or "local"),
        "secondTtsHost": str(settings.get("secondTtsHost") or ""),
        "autoEmoji": bool(settings.get("autoEmoji", True)),
        "webSearch": bool(settings.get("webSearch", False)),
        "twoPlayerMode": bool(settings.get("twoPlayerMode", False)),
        "twoOnlyMode": bool(settings.get("twoOnlyMode", False)),
        "emojiStyle": str(settings.get("emojiStyle") or ""),
        "emojiCustom": str(settings.get("emojiCustom") or ""),
    }


def write_session_settings(settings: dict) -> dict:
    """グローバル設定（会話ログを含まない）を latest_session.json へ保存する。"""
    PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
    profile = {
        "version": 3,
        "savedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "settings": normalize_session_settings(settings),
    }
    SESSION_PROFILE_PATH.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return profile


def _prior_mode_map(history_file: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """既存 history.json の (role, content) → (mode, speaker) を読み出す。

    自動保存はクライアントから来た全履歴で丸ごと上書きするが、クライアントは
    手作業/再構築で付けた mode/speaker を送り返さない場合がある。それらを毎回の
    保存で失わないよう、ディスク上の既存注釈を内容一致で引き継ぐためのマップ。
    """
    if not history_file.exists():
        return {}
    try:
        data = json.loads(history_file.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    history = data.get("history") if isinstance(data, dict) else None
    result: dict[tuple[str, str], tuple[str, str]] = {}
    if isinstance(history, list):
        for item in history:
            if not isinstance(item, dict):
                continue
            key = (str(item.get("role") or ""), str(item.get("content") or "").strip())
            mode = str(item.get("mode") or "").strip()
            speaker = str(item.get("speaker") or "").strip()
            ts = str(item.get("ts") or "").strip()
            if mode in {"normal", "two_only"} or speaker or ts:
                result[key] = (
                    mode if mode in {"normal", "two_only"} else "",
                    speaker,
                    ts,
                )
    return result


def _existing_history_keys(history_file: Path) -> set[tuple[str, str]]:
    """既存 history.json に入っている (role, content) の集合を返す。

    「今回追加された新しいターン」を見分けるために使う。クライアントは全履歴を
    送り返してくるので、ディスク上に無いエントリだけが新規ターンである。
    """
    if not history_file.exists():
        return set()
    try:
        data = json.loads(history_file.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return set()
    history = data.get("history") if isinstance(data, dict) else data
    if not isinstance(history, list):
        return set()
    return {
        (str(item.get("role") or ""), str(item.get("content") or "").strip())
        for item in history
        if isinstance(item, dict)
    }


def write_character_history(char_id: str, entries: list[dict]) -> Path:
    """1 キャラ分の会話ログを profiles/sessions/<charId>/history.json へ書き出す。"""
    safe_id = safe_character_id(char_id)
    char_dir = SESSION_HISTORY_ROOT / safe_id
    char_dir.mkdir(parents=True, exist_ok=True)
    history_file = char_dir / "history.json"

    known_keys = _existing_history_keys(history_file)
    sanitized = sanitize_history(entries)
    # クライアントが mode/speaker を送り返さなくても、ディスク上の既存注釈を内容一致で
    # 復元する（手動注釈が自動保存で消えるのを防ぐ）。新規ターンは既存に無いので、
    # 会話モードは memory.sqlite3 側（save_memory）が確定値を持つ＝ここでは normal 相当。
    prior = _prior_mode_map(history_file)
    if prior:
        for entry in sanitized:
            key = (str(entry.get("role") or ""), str(entry.get("content") or "").strip())
            saved = prior.get(key)
            if not saved:
                continue
            saved_mode, saved_speaker, saved_ts = saved
            if saved_mode and "mode" not in entry:
                entry["mode"] = saved_mode
            if saved_speaker and "speaker" not in entry:
                entry["speaker"] = saved_speaker
            if saved_ts and "ts" not in entry:
                entry["ts"] = saved_ts

    # 新規ターンにだけ現在時刻を刻む。「いつ話したか」を history.json 側にも残すことで、
    # 履歴から DB を作り直しても時刻が失われない（tools/rebuild_rag_from_history.py）。
    # 既にディスクにあるエントリには触らない: クライアントは毎回全履歴を送り返すため、
    # 無条件に刻むと ts の無い過去ターン全部が「今日」に書き換わって時系列が壊れる。
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for entry in sanitized:
        if entry.get("ts"):
            continue
        key = (str(entry.get("role") or ""), str(entry.get("content") or "").strip())
        if key not in known_keys:
            entry["ts"] = stamp

    payload = {
        "version": 2,
        "savedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "characterId": str(char_id),
        "history": sanitized,
    }
    history_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return history_file


def read_character_histories() -> dict[str, list[dict]]:
    """profiles/sessions/*/history.json をすべて読み込みキャラ別ログのマップを返す。"""
    result: dict[str, list[dict]] = {}
    if not SESSION_HISTORY_ROOT.exists():
        return result
    for char_dir in sorted(SESSION_HISTORY_ROOT.iterdir()):
        if not char_dir.is_dir():
            continue
        history_file = char_dir / "history.json"
        if not history_file.exists():
            continue
        try:
            data = json.loads(history_file.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        char_id = str(data.get("characterId") or char_dir.name).strip() or char_dir.name
        result[char_id] = sanitize_history(data.get("history"))
    return result


def save_session_profile(payload: dict) -> dict:
    settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
    active_main = str(settings.get("activeMainCharacterId") or "rinon")
    histories = sanitize_histories(payload.get("histories"))
    if not histories and isinstance(payload.get("history"), list):
        # 旧クライアント互換: 単一 history はアクティブなメインキャラのログとして扱う。
        legacy = sanitize_history(payload.get("history"))
        if legacy:
            histories = {active_main: legacy}
    profile = write_session_settings(settings)
    for char_id, entries in histories.items():
        write_character_history(char_id, entries)
    profile["histories"] = histories
    return profile


def load_session_profile() -> dict:
    settings: dict = {}
    legacy_history: object = None
    legacy_histories: object = None
    profile_exists = SESSION_PROFILE_PATH.exists()
    if profile_exists:
        raw = json.loads(SESSION_PROFILE_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("saved profile is invalid")
        settings = raw.get("settings") if isinstance(raw.get("settings"), dict) else {}
        # 旧フォーマットでは会話ログが settings と同じファイルに同居していた。
        if isinstance(raw.get("histories"), dict):
            legacy_histories = raw.get("histories")
        elif isinstance(raw.get("history"), list):
            legacy_history = raw.get("history")
    active_main = str(settings.get("activeMainCharacterId") or "rinon")

    # 新フォーマット: キャラ別フォルダから会話ログを読み込む。
    histories = read_character_histories()

    # 旧ログの自動振り分け: フォルダ側が空で、旧ファイルにログが残っている場合のみ移行する。
    migrated = False
    if not histories:
        if isinstance(legacy_histories, dict):
            histories = sanitize_histories(legacy_histories)
            migrated = bool(histories)
        elif isinstance(legacy_history, list):
            legacy = sanitize_history(legacy_history)
            if legacy:
                # 単一ログは保存時のアクティブなメインキャラのログとして振り分ける。
                histories = {active_main: legacy}
                migrated = True

    if migrated:
        for char_id, entries in histories.items():
            write_character_history(char_id, entries)
        # 移行済みログを latest_session.json から取り除き、以降はフォルダ側を正とする。
        if profile_exists:
            write_session_settings(settings)

    return {
        "exists": profile_exists or bool(histories),
        "path": str(SESSION_PROFILE_PATH),
        "settings": settings,
        "histories": histories,
        "history": histories.get(active_main, []),
    }


def sanitize_reference_path(value: object, fallback: Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return fallback
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (APP_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if candidate.suffix.lower() not in ALLOWED_REFERENCE_EXTENSIONS:
        return fallback
    if not candidate.exists():
        return fallback
    return candidate


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def irodori_python_path() -> Path:
    override = os.environ.get("IRODORI_PYTHON", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        return IRODORI_ROOT / ".venv" / "Scripts" / "python.exe"
    return IRODORI_ROOT / ".venv" / "bin" / "python"


def is_auto_runtime_value(value: str) -> bool:
    return str(value or "").strip().lower() in {"", "auto", "default"}


def default_irodori_runtime_device() -> str:
    from irodori_tts.inference_runtime import default_runtime_device

    return default_runtime_device()


def irodori_precision_for_device(device: str, requested: str) -> str:
    if not is_auto_runtime_value(requested):
        return str(requested).strip().lower()
    from irodori_tts.inference_runtime import list_available_runtime_precisions

    choices = list_available_runtime_precisions(device)
    device_type = str(device).split(":", 1)[0].lower()
    if device_type in {"cuda", "xpu"} and "bf16" in choices:
        return "bf16"
    return choices[0] if choices else "fp32"


def irodori_runtime_settings() -> dict[str, str]:
    model_device = IRODORI_MODEL_DEVICE
    codec_device = IRODORI_CODEC_DEVICE
    if is_auto_runtime_value(model_device) or is_auto_runtime_value(codec_device):
        default_device = default_irodori_runtime_device()
        if is_auto_runtime_value(model_device):
            model_device = default_device
        if is_auto_runtime_value(codec_device):
            codec_device = default_device
    return {
        "modelDevice": str(model_device).strip().lower(),
        "modelPrecision": irodori_precision_for_device(model_device, IRODORI_MODEL_PRECISION),
        "codecDevice": str(codec_device).strip().lower(),
        "codecPrecision": irodori_precision_for_device(codec_device, IRODORI_CODEC_PRECISION),
    }


def quiet_irodori_watermark_warnings() -> None:
    import irodori_tts.inference_runtime as inference_runtime

    base_watermarker = inference_runtime.SilentCipherWatermarker
    if getattr(base_watermarker, "__rinon_quiet__", False):
        return

    class QuietSilentCipherWatermarker(base_watermarker):
        __rinon_quiet__ = True

        def __init__(self, *args, **kwargs) -> None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"`torch\.nn\.utils\.weight_norm` is deprecated in favor of `torch\.nn\.utils\.parametrizations\.weight_norm`.*",
                    category=FutureWarning,
                )
                super().__init__(*args, **kwargs)

        def encode_batch(self, audios: list, *, sample_rate: int) -> list:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"An output with one or more elements was resized since it had shape \[\].*",
                    category=UserWarning,
                )
                with contextlib.redirect_stdout(FilteredIrodoriStdout(sys.stdout)):
                    return super().encode_batch(audios, sample_rate=sample_rate)

    inference_runtime.SilentCipherWatermarker = QuietSilentCipherWatermarker


def normalize_remote_tts_url(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if not re.match(r"^https?://", text, re.IGNORECASE):
        text = f"http://{text}"
    parsed = urlparse(text)
    if not parsed.netloc:
        return ""
    netloc = parsed.netloc
    try:
        has_port = parsed.port is not None
    except ValueError:
        has_port = False
    if not has_port and ":" not in netloc:
        netloc = f"{netloc}:{LUVIA_REMOTE_DEFAULT_PORT}"
    path = parsed.path.rstrip("/")
    if path.endswith("/synthesize"):
        path = path[: -len("/synthesize")]
    return f"{parsed.scheme}://{netloc}{path}".rstrip("/")


def remote_luvia_enabled(remote_url: str = "") -> bool:
    return bool(
        normalize_remote_tts_url(remote_url)
        or LUVIA_REMOTE_TTS_URL
        or (LUVIA_REMOTE_TTS_HOST and LUVIA_REMOTE_IRODORI_ROOT)
    )


def environment_diagnostics() -> dict:
    irodori_python = irodori_python_path()
    irodori_pyproject = IRODORI_ROOT / "pyproject.toml"
    lm_models = get_models()
    return {
        "appRoot": str(APP_ROOT),
        "irodoriRoot": str(IRODORI_ROOT),
        "irodoriRootExists": IRODORI_ROOT.exists(),
        "irodoriPython": str(irodori_python),
        "irodoriPythonExists": irodori_python.exists(),
        "irodoriProjectExists": irodori_pyproject.exists(),
        "gitExists": command_exists("git"),
        "uvExists": command_exists("uv"),
        "irodoriModelDevice": IRODORI_MODEL_DEVICE,
        "irodoriModelPrecision": IRODORI_MODEL_PRECISION,
        "irodoriCodecDevice": IRODORI_CODEC_DEVICE,
        "irodoriCodecPrecision": IRODORI_CODEC_PRECISION,
        "lmStudioUrl": LM_STUDIO_URL,
        "lmStudioReady": bool(lm_models),
        "models": lm_models,
        # 環境変数から解決した既定モデル。プルダウンに並ぶ ID の表記へ寄せて返す。
        "preferredModel": preferred_model_option(lm_models),
        # プルダウンで選ばせてよいモデル（対象外は表示だけして選べなくする）。
        "supportedModels": [name for name in lm_models if model_is_supported(name)],
        "remoteLuviaEnabled": remote_luvia_enabled(),
        "remoteLuviaUrl": LUVIA_REMOTE_TTS_URL,
        "remoteLuviaHost": LUVIA_REMOTE_TTS_HOST,
        "remoteLuviaRoot": LUVIA_REMOTE_IRODORI_ROOT,
        "remoteLuviaReference": LUVIA_REMOTE_REF_WAV,
        "referenceExists": IRODORI_REF_WAV.exists(),
        "luviaReferenceExists": LUVIA_REF_WAV.exists(),
    }


def save_reference_audio(payload: dict) -> dict:
    slot = "second" if str(payload.get("slot") or "") == "second" else "main"
    character_id = sanitize_character_id(payload.get("characterId"), "rinon")
    original_name = Path(str(payload.get("name") or "reference.wav")).name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_REFERENCE_EXTENSIONS:
        raise ValueError("reference audio must be wav, mp3, flac, m4a, ogg, or aac")
    encoded = str(payload.get("dataBase64") or "")
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    audio_bytes = base64.b64decode(encoded, validate=True)
    if not audio_bytes:
        raise ValueError("reference audio is empty")
    if len(audio_bytes) > 80 * 1024 * 1024:
        raise ValueError("reference audio is too large")

    out_dir = CHARACTER_ROOT / character_id / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^0-9A-Za-z_-]+", "_", Path(original_name).stem).strip("_")[:48]
    safe_stem = safe_stem or "reference"
    file_name = f"{slot}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{safe_stem}{suffix}"
    out_path = out_dir / file_name
    out_path.write_bytes(audio_bytes)
    return {
        "ok": True,
        "characterId": character_id,
        "slot": slot,
        "path": str(out_path),
        "url": character_url(character_id, "reference", file_name),
        "name": file_name,
        "size": out_path.stat().st_size,
    }


def remote_ref_for_luvia(reference_wav: Path) -> str:
    if not (LUVIA_REMOTE_TTS_HOST and LUVIA_REMOTE_IRODORI_ROOT and LUVIA_REMOTE_REF_WAV):
        raise RuntimeError("Remote Luvia TTS is not configured")
    if reference_wav.resolve() == LUVIA_REF_WAV.resolve():
        return LUVIA_REMOTE_REF_WAV
    cache_key = str(reference_wav.resolve())
    if cache_key in Luvia_remote_ref_cache:
        return Luvia_remote_ref_cache[cache_key]
    request_id = f"ref_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}{reference_wav.suffix.lower()}"
    remote_refs = rf"{LUVIA_REMOTE_IRODORI_ROOT}\remote_refs"
    remote_path = rf"{remote_refs}\{request_id}"

    def run_command(args: list[str], timeout: int = 60) -> None:
        completed = subprocess.run(
            args,
            cwd=str(APP_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(detail or f"command failed: {' '.join(args)}")

    run_command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            LUVIA_REMOTE_TTS_HOST,
            f'cmd /c if not exist "{remote_refs}" mkdir "{remote_refs}"',
        ],
        timeout=30,
    )
    remote_scp = remote_path.replace("\\", "/")
    run_command(["scp", "-q", str(reference_wav), f"{LUVIA_REMOTE_TTS_HOST}:{remote_scp}"], timeout=90)
    Luvia_remote_ref_cache[cache_key] = remote_path
    return remote_path


def save_current_audio(payload: dict) -> dict:
    url = str(payload.get("url") or "").strip()
    parsed = urlparse(url)
    rel_url = parsed.path if parsed.scheme or parsed.netloc else url
    if not rel_url.startswith("/generated/"):
        raise ValueError("only generated app audio can be saved")
    source_name = Path(unquote(rel_url)).name
    source_path = (STATIC_ROOT / "generated" / source_name).resolve()
    generated_root = (STATIC_ROOT / "generated").resolve()
    if not str(source_path).startswith(str(generated_root)) or not source_path.exists():
        raise FileNotFoundError("audio file was not found")

    # 保存音声もキャラクターごとのフォルダに振り分ける（saved_audio/<charId>/...）。
    char_dir_name = safe_character_id(payload.get("characterId"))
    audio_dir = SAVED_AUDIO_ROOT / char_dir_name
    audio_dir.mkdir(parents=True, exist_ok=True)
    label = re.sub(r"[^0-9A-Za-z_-]+", "_", str(payload.get("label") or "rinon").strip())
    label = label.strip("_")[:40] or "rinon"
    saved_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{label}_{source_path.name}"
    saved_path = audio_dir / saved_name
    shutil.copy2(source_path, saved_path)
    return {
        "ok": True,
        "path": str(saved_path),
        "url": f"/saved_audio/{char_dir_name}/{saved_name}",
        "name": saved_name,
        "size": saved_path.stat().st_size,
    }


def generated_name_from_saved(file_name: str) -> str:
    """旧保存名 ``<日付>_<時刻>_<label>_<生成ファイル名>`` から生成ファイル名を取り出す。

    保存音声は ``save_current_audio`` が ``f"{ts}_{label}_{source_path.name}"`` で命名する。
    ``ts`` は ``%Y%m%d_%H%M%S``（アンダースコア1個）なので、先頭3トークン
    （日付・時刻・label）を除いた残りが元の生成ファイル名（例: ``reply_..._combined.wav``）。
    """
    parts = file_name.split("_", 3)
    return parts[3] if len(parts) >= 4 else ""


def legacy_audio_label(file_name: str) -> str:
    """旧保存名の label トークン（固定値 rinon / 2p 等）を取り出す。照合失敗時の保険用。"""
    parts = file_name.split("_")
    return parts[2] if len(parts) >= 3 else ""


def build_audio_owner_index() -> dict[str, str]:
    """各キャラのログに登録された音声ファイル名 → キャラ ID の対応表を作る。

    ログの ``display.audioUrl`` は ``/generated/<生成ファイル名>`` を指す。この生成
    ファイル名は保存音声のファイル名末尾にもそのまま含まれるため、これを突き合わせれば
    旧保存音声がどのキャラの会話で生成されたものかを確定できる。
    """
    index: dict[str, str] = {}
    for char_id, entries in read_character_histories().items():
        for entry in entries:
            display = entry.get("display") if isinstance(entry, dict) else None
            if not isinstance(display, dict):
                continue
            url = str(display.get("audioUrl") or "")
            if not url:
                continue
            base = Path(urlparse(url).path or url).name
            if base:
                index.setdefault(base, char_id)
    return index


def migrate_legacy_audio() -> int:
    """saved_audio/ 直下の旧形式ファイルを、新形式（キャラ別フォルダ）へ複製する。

    振り分け先は、キャラ別ログに登録された音声ファイル名との照合で確定する（確実）。
    照合できないファイルのみ、旧ファイル名の label をフォールダ名にフォールバックする。
    元ファイルはフォールバック配信用にその場へ残し、新形式の場所へ同名で複製する。
    複製済みならスキップするため、起動のたびに呼んでも安全（冪等）。
    """
    if not SAVED_AUDIO_ROOT.exists():
        return 0
    owner_index = build_audio_owner_index()
    copied = 0
    for entry in sorted(SAVED_AUDIO_ROOT.iterdir()):
        if not entry.is_file():
            continue  # 既に振り分け済みのサブフォルダは対象外。
        gen_name = generated_name_from_saved(entry.name)
        # まずログ照合でキャラを確定。見つからなければ label をフォールバックに使う。
        bucket = owner_index.get(gen_name) or legacy_audio_label(entry.name)
        target_dir = SAVED_AUDIO_ROOT / safe_character_id(bucket)
        target_path = target_dir / entry.name
        if target_path.exists():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry, target_path)
        copied += 1
    return copied


def run_startup_migrations() -> None:
    """起動時に、旧形式のログ・音声ファイルを新形式へ移行する。"""
    try:
        # 旧形式の会話ログ（latest_session.json 内）をキャラ別フォルダへ移行する。
        load_session_profile()
    except Exception as exc:  # 移行失敗は致命的ではないため起動は継続する。
        print(f"[startup] session log migration skipped: {exc}", flush=True)
    try:
        copied = migrate_legacy_audio()
        if copied:
            print(f"[startup] migrated {copied} legacy audio file(s) into per-character folders", flush=True)
    except Exception as exc:
        print(f"[startup] audio migration skipped: {exc}", flush=True)


def stop_irodori_ui_processes() -> list[dict[str, str | int]]:
    current_pid = os.getpid()
    irodori_root_text = str(IRODORI_ROOT).lower()
    script = r"""
$ports = @(7861)
$portPids = @()
foreach ($port in $ports) {
  Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { $portPids += [int]$_ }
}
Get-CimInstance Win32_Process |
  Where-Object {
    $_.CommandLine -and (
      ($_.CommandLine -like '*Irodori-TTS*' -and $_.CommandLine -match '(gradio|voicedesign|base_ui|infer\.py|uv run|python)') -or
      ($portPids -contains [int]$_.ProcessId)
    )
  } |
  Select-Object ProcessId, Name, CommandLine |
  ConvertTo-Json -Compress
"""
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            cwd=str(APP_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except Exception as exc:
        return [{"pid": 0, "name": "scan failed", "detail": str(exc)}]
    if completed.returncode != 0:
        return [{"pid": 0, "name": "scan failed", "detail": (completed.stderr or completed.stdout).strip()}]
    raw = (completed.stdout or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [{"pid": 0, "name": "scan parse failed", "detail": raw[:500]}]
    items = data if isinstance(data, list) else [data]
    stopped: list[dict[str, str | int]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        pid = int(item.get("ProcessId") or 0)
        command_line = str(item.get("CommandLine") or "")
        if not pid or pid == current_pid:
            continue
        if irodori_root_text not in command_line.lower() and "7861" not in command_line:
            continue
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                cwd=str(APP_ROOT),
                capture_output=True,
                text=True,
                timeout=15,
            )
            stopped.append({"pid": pid, "name": str(item.get("Name") or ""), "detail": "stopped"})
        except Exception as exc:
            stopped.append({"pid": pid, "name": str(item.get("Name") or ""), "detail": str(exc)})
    return stopped


def shutdown_app_server(server: ThreadingHTTPServer) -> None:
    def worker() -> None:
        time.sleep(0.35)
        try:
            server.shutdown()
            server.server_close()
        finally:
            time.sleep(0.35)
            os._exit(0)

    threading.Thread(target=worker, daemon=True).start()


def build_emoji_choice_prompt() -> str:
    items = load_emoji_items()
    lines = [f"{item['emoji']} = {item['label']} / {item['description']}" for item in items]
    return "\n".join(lines)


def parse_lmstudio_reply(raw: str, allowed_emojis: set[str]) -> tuple[str, str]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
        reply = str(data.get("text") or data.get("reply") or "").strip()
        emoji = str(data.get("emoji") or "").strip()
        if reply:
            return reply, emoji if emoji in allowed_emojis else ""
    except Exception:
        pass
    return raw.strip(), ""


def compose_caption(base_caption: str, style: str) -> str:
    """基底 TTS Caption（キャラの基本的なしゃべり方）を先頭に、感情/口調を末尾に連結する。

    感情指示（``style``）が空なら基底 caption だけを返し、現状と同じ挙動になる。
    """
    base = str(base_caption or "").strip()
    emotion = str(style or "").strip()
    if not emotion:
        return base
    if not base:
        return emotion
    separator = "" if base[-1] in "。.．!?！？…、,，・ " else " "
    return f"{base}{separator} {emotion}".replace("  ", " ").strip()


def _repair_segment_json(text: str) -> str:
    """LM が出しがちな軽微な JSON 崩れを補修する。

    典型例: ``emoji`` のキー名を落として ``,""`` になる／末尾カンマ。
    """
    repaired = text
    # ,"" のようにキー名の無い空値メンバーを除去（オブジェクト末尾・中間の両方）。
    repaired = re.sub(r',\s*""\s*(?=[}\]])', "", repaired)
    repaired = re.sub(r',\s*""\s*,', ",", repaired)
    # 末尾カンマ（, } / , ]）の除去。
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


def _json_str_field(obj_text: str, key: str) -> str:
    """オブジェクト断片から ``"key":"value"`` の文字列値を取り出す（エスケープ考慮）。"""
    match = re.search(r'"' + re.escape(key) + r'"\s*:\s*"((?:\\.|[^"\\])*)"', obj_text)
    if not match:
        return ""
    try:
        return json.loads('"' + match.group(1) + '"')
    except Exception:
        return match.group(1)


def _extract_segments_regex(text: str, allowed_emojis: set[str]) -> list[dict[str, str]]:
    """壊れた JSON でも、キー名ベースの正規表現で segments を救済抽出する。"""
    segments: list[dict[str, str]] = []
    for obj in re.findall(r"\{[^{}]*\}", text):
        if '"text"' not in obj:
            continue
        seg_text = strip_irodori_style_marks(_json_str_field(obj, "text") or _json_str_field(obj, "reply"))
        if not seg_text:
            continue
        style = (_json_str_field(obj, "style") or _json_str_field(obj, "caption")).strip()
        emoji = _json_str_field(obj, "emoji").strip()
        if emoji not in allowed_emojis:
            emoji = ""
        segments.append({"text": seg_text, "style": style, "emoji": emoji})
    return segments


def parse_lmstudio_segments(raw: str, allowed_emojis: set[str]) -> list[dict[str, str]]:
    """LM 返答から ``{"segments":[{text, style, emoji}...]}`` を取り出す。

    LM は長め/複雑な返答でしばしば壊れた JSON（例: ``emoji`` キー名の欠落 ``,""``）を
    返すため、厳密パース→軽微修復→キー名正規表現の順で頑健に抽出する。取得できなければ
    空リストを返し、呼び出し側は従来の「返答全体で1感情・1caption」へフォールバックする。
    ``emoji`` は irodori パレット（``allowed_emojis``）に含まれるものだけを採用する。
    """
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    data = None
    for candidate in (text, _repair_segment_json(text)):
        try:
            data = json.loads(candidate)
            break
        except Exception:
            data = None

    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        segments: list[dict[str, str]] = []
        for item in data["segments"]:
            if not isinstance(item, dict):
                continue
            seg_text = strip_irodori_style_marks(str(item.get("text") or item.get("reply") or ""))
            if not seg_text:
                continue
            style = str(item.get("style") or item.get("caption") or "").strip()
            emoji = str(item.get("emoji") or "").strip()
            if emoji not in allowed_emojis:
                emoji = ""
            segments.append({"text": seg_text, "style": style, "emoji": emoji})
        if segments:
            return segments

    # 厳密/修復パースが失敗、または有効な segment が0件 → 正規表現で救済抽出する。
    return _extract_segments_regex(text, allowed_emojis)


def strip_irodori_style_marks(text: str) -> str:
    cleaned = str(text or "")
    emojis = sorted((item["emoji"] for item in load_emoji_items()), key=len, reverse=True)
    for emoji in emojis:
        cleaned = cleaned.replace(emoji, "")
    return re.sub(r"\s+", " ", cleaned).strip()


def sanitize_no_dialogue_reply(text: str) -> str:
    allowed_fragments = (
        "好き",
        "大好き",
        "感じる",
        "感じちゃう",
        "だめ",
        "だめえ",
        "だめぇ",
        "だめっ",
        "もうだめ",
        "我慢できない",
        "おかしくなっちゃう",
        "いかせて",
        "いかせてぇ",
        "無理",
    )
    hard_banned_fragments = (
        "きみ",
        "君",
        "あなた",
        "あんた",
        "こっち",
        "そこ",
        "ここ",
        "反応",
        "可愛い",
        "かわいい",
        "止め",
        "してほしい",
        "ほしい",
        "して",
        "ほら",
        "ねぇ",
        "ねえ",
        "混ぜて",
        "来て",
        "見て",
        "？",
        "?",
    )
    parts = re.split(r"(?<=[。！？!?])\s*|\n+", str(text or ""))
    kept: list[str] = []
    for part in parts:
        candidate = strip_irodori_style_marks(part).strip()
        if not candidate:
            continue
        allowed_hit = any(fragment in candidate for fragment in allowed_fragments)
        if any(fragment in candidate for fragment in hard_banned_fragments):
            continue
        if re.search(r"[一-龯々〆ヵヶ]", candidate) and not allowed_hit:
            continue
        if len(candidate) > (64 if allowed_hit else 48):
            continue
        kept.append(candidate)
    cleaned = " ".join(kept)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) < 6:
        return "……っ、はぁ……。ん……っ。……ふぅ……。"
    return cleaned


def reply_style_for_length(reply_length: str) -> tuple[str, int, int]:
    mode = str(reply_length or "normal").strip().lower()
    if mode == "long":
        return "返答は6から10文くらいまで使って、自然な会話調で少し詳しく答えてください。", 1200, 10
    if mode == "short":
        return "返答は1から2文を基本にしてください。", 360, 3
    return "返答は3から5文くらいまで使って、自然な会話調で答えてください。", 720, 6


def tts_duration_scale_for_rate(value: object) -> float:
    mode = str(value or "normal").strip().lower()
    if mode == "fast":
        return 0.86
    return 1.0


def trim_messages_for_context(messages: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    limit = max(1000, int(limit or DEFAULT_CONTEXT_LIMIT))
    kept: list[dict[str, str]] = []
    used = 0
    for item in reversed(messages):
        content = str(item.get("content") or "")
        cost = len(content) + 32
        if kept and used + cost > limit:
            break
        kept.append(item)
        used += cost
    return list(reversed(kept))


def message_context_cost(messages: list[dict[str, str]]) -> int:
    return sum(len(str(item.get("content") or "")) + 32 for item in messages)


def compact_text(value: str, limit: int = 110) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def summarize_old_messages(messages: list[dict[str, str]], limit: int = LM_SUMMARY_CHAR_LIMIT) -> str:
    if not messages:
        return ""
    topic_lines: list[str] = []
    turn_lines: list[str] = []
    for item in messages:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if content.startswith(("お題:", "次のお題:")):
            topic_lines.append(compact_text(content, 90))
        role = "ユーザー" if item.get("role") == "user" else "相手"
        if content.startswith(("リノン:", "ルヴィア:")) and ":" in content:
            role, content = content.split(":", 1)
            role = role.strip() or "相手"
            content = content.strip()
        turn_lines.append(f"{role}: {compact_text(content, 95)}")

    selected: list[str] = []
    if topic_lines:
        selected.append("過去のお題: " + " / ".join(topic_lines[-5:]))
    if turn_lines:
        selected.append("古い会話の圧縮ログ:")
        selected.extend(turn_lines[-14:])
    summary = "\n".join(selected)
    if len(summary) > limit:
        summary = "… " + summary[-limit:].lstrip()
    return summary


def compact_messages_for_context(
    messages: list[dict[str, str]],
    limit: int,
) -> tuple[list[dict[str, str]], dict[str, int | bool]]:
    requested_limit = max(1000, int(limit or DEFAULT_CONTEXT_LIMIT))
    effective_limit = min(requested_limit, max(1000, LM_COMPACT_CONTEXT_LIMIT))
    recent_count = max(4, LM_RECENT_MESSAGE_COUNT)
    full_cost = message_context_cost(messages)
    if full_cost <= effective_limit and len(messages) <= recent_count:
        trimmed = trim_messages_for_context(messages, effective_limit)
        return trimmed, {
            "full": full_cost,
            "sent": message_context_cost(trimmed),
            "limit": requested_limit,
            "effectiveLimit": effective_limit,
            "compacted": False,
            "recentMessages": len(trimmed),
        }

    recent = messages[-recent_count:]
    older = messages[:-recent_count]
    summary = summarize_old_messages(older)
    compacted: list[dict[str, str]] = []
    if summary:
        compacted.append(
            {
                "role": "user",
                "content": (
                    "以下は古い会話の要約です。細部よりも、現在の関係性、進行中のお題、"
                    "直前までの流れを保つための参考として扱ってください。\n"
                    f"{summary}"
                ),
            }
        )
    compacted.extend(recent)
    trimmed = trim_messages_for_context(compacted, effective_limit)
    if summary and compacted and (not trimmed or trimmed[0] != compacted[0]):
        compacted[0]["content"] = compacted[0]["content"][-max(400, effective_limit // 3) :]
        trimmed = trim_messages_for_context(compacted, effective_limit)
    return trimmed, {
        "full": full_cost,
        "sent": message_context_cost(trimmed),
        "limit": requested_limit,
        "effectiveLimit": effective_limit,
        "compacted": bool(summary),
        "recentMessages": min(len(recent), len(trimmed)),
    }


def strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = html_unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_search_url(value: str) -> str:
    url = html_unescape(str(value or ""))
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return url


def web_search(query: str, limit: int = 3) -> list[dict[str, str]]:
    search_query = re.sub(r"\s+", " ", str(query or "")).strip()
    if not search_query:
        return []
    url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(search_query[:220])}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
        },
    )
    with urllib.request.urlopen(request, timeout=WEB_SEARCH_TIMEOUT) as res:
        html = res.read().decode("utf-8", errors="replace")
    results: list[dict[str, str]] = []
    anchor_pattern = re.compile(r"<a(?P<attrs>[^>]*)>(?P<title>.*?)</a>", re.IGNORECASE | re.DOTALL)
    anchors = [match for match in anchor_pattern.finditer(html) if "result-link" in match.group("attrs")]
    for index, match in enumerate(anchors):
        attrs = match.group("attrs")
        href_match = re.search(r"href=['\"](?P<href>[^'\"]+)['\"]", attrs, re.IGNORECASE)
        if not href_match:
            continue
        title = strip_html(match.group("title"))
        href = normalize_search_url(href_match.group("href"))
        tail_end = anchors[index + 1].start() if index + 1 < len(anchors) else len(html)
        tail = html[match.end() : tail_end]
        snippet_match = re.search(
            r"<td[^>]*result-snippet[^>]*>(?P<snippet>.*?)</td>",
            tail,
            re.IGNORECASE | re.DOTALL,
        )
        snippet = strip_html(snippet_match.group("snippet")) if snippet_match else ""
        if title and href:
            results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


WEB_INTENT_TERMS = (
    "評判",
    "口コミ",
    "レビュー",
    "感想",
    "映画",
    "新作",
    "公開",
    "公開日",
    "監督",
    "声優",
    "キャスト",
    "制作",
    "予告",
    "配信",
    "あらすじ",
    "ネタバレ",
)
WEB_QUERY_STOP_TERMS = {"リノン", "ルヴィア", "ユーザー", "お題", "会話", "自動会話"}


def extract_web_query_terms(text: str, include_intents: bool = True) -> list[str]:
    source = str(text or "")
    terms: list[str] = []
    for quoted in re.findall(r"[「『\"]([^」』\"]{2,40})[」』\"]", source):
        terms.append(quoted)
    terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9._+-]{1,}", source))
    terms.extend(re.findall(r"[ァ-ヶー]{2,}", source))
    if include_intents:
        for term in WEB_INTENT_TERMS:
            if term in source:
                terms.append(term)
    cleaned: list[str] = []
    seen: set[str] = set()
    for term in terms:
        value = re.sub(r"\s+", " ", term).strip("。、，,.!?！？:：;；()（）[]【】")
        if len(value) < 2 or value in WEB_QUERY_STOP_TERMS or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned


def build_continuous_web_query(user_text: str, history: list[dict[str, str]]) -> str:
    current_terms = extract_web_query_terms(user_text, include_intents=True)
    inherited: list[str] = []
    for item in reversed(history[-8:]):
        content = str(item.get("content") or "")
        if ":" in content and content.split(":", 1)[0] in {"リノン", "ルヴィア"}:
            content = content.split(":", 1)[1]
        for term in extract_web_query_terms(content, include_intents=False):
            if term not in inherited:
                inherited.append(term)
        if len(inherited) >= 6:
            break
    combined: list[str] = []
    for term in [*inherited, *current_terms]:
        if term not in combined:
            combined.append(term)
    if combined:
        return " ".join(combined[:8])
    return re.sub(r"\s+", " ", str(user_text or "")).strip()


def format_web_results(query: str, results: list[dict[str, str]]) -> str:
    if not results:
        return ""
    lines = [
        f'Web検索結果です。検索語: "{compact_text(query, 120)}"',
        "現在までの会話コンテキスト、キャラ設定、直前の発言を優先してください。",
        "検索結果は補助情報として必要な場合だけ使い、検索結果にない事実は断定しないでください。",
    ]
    for index, item in enumerate(results, start=1):
        lines.append(
            f"{index}. {compact_text(item.get('title', ''), 100)}\n"
            f"URL: {item.get('url', '')}\n"
            f"概要: {compact_text(item.get('snippet', ''), 180)}"
        )
    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """日本語混在テキストのトークン数をざっくり見積もる（安全側＝多めに寄せる）。

    正確なトークナイザは OpenAI 互換 API 経由では引けないので概算にする。係数は
    gemma-4 の prompt_tokens 実測から取った（tools/ ではなく手元計測）:
      ・純日本語の散文        4000字 → 2616tok（0.65 tok/字）
      ・年表風（日付＋番号）  6479字 → 4695tok
      ・台帳風（日付＋番号）  4679字 → 3855tok（0.82 tok/字）
      ・英語混在              2800字 → 1417tok
    数字は 1 文字ずつ独立したトークンになりやすく、記憶ブロックは日付と番号だらけ
    なので、数字を文字数と同じだけ数えるのが要点（ここを甘く見ると想定より実トークン
    が膨らみ、文脈超過の 400 で弾かれる）。予算を外すとそのまま空返答になるため、
    どの実測サンプルでも est ≧ actual になる係数を選んでいる。
    """
    body = str(text or "")
    if not body:
        return 0
    digits = 0
    other_ascii = 0
    non_ascii = 0
    for ch in body:
        if ch.isdigit():
            digits += 1
        elif ord(ch) < 128:
            other_ascii += 1
        else:
            non_ascii += 1
    return int(digits * 1.05 + other_ascii * 0.5 + non_ascii * 0.85) + 1


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """messages 配列ぶんの概算トークン。役割・区切りのテンプレート分を 1 通 8tok 見る。"""
    return sum(estimate_tokens(str(item.get("content") or "")) + 8 for item in messages)


def _model_id_matches(requested: str, listed: str) -> bool:
    """モデル名の書き方の違い（リポジトリ付き・量子化サフィックス・.gguf）を吸収して比べる。

    例: `lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12b-it-Q6_K.gguf` と
    `gemma-4-12b-it@q6_k` を同一とみなす。
    """

    def norm(value: str) -> str:
        text = str(value or "").strip().lower().rsplit("/", 1)[-1]
        if text.endswith(".gguf"):
            text = text[: -len(".gguf")]
        text = text.split("@", 1)[0]
        return re.sub(r"[^a-z0-9]", "", text)

    left, right = norm(requested), norm(listed)
    if not left or not right:
        return False
    return left == right or left.startswith(right) or right.startswith(left)


def _lm_server_root() -> str:
    """OpenAI 互換の ``/v1`` を除いた推論サーバのルート URL を返す。

    ``/api/v0/models``（LM Studio）や ``/slots``（llama-server）はルート直下にあり、
    ``/v1`` 配下には無い。両方を叩くのでここで 1 箇所にまとめる。
    """
    if LM_STUDIO_URL.endswith("/v1"):
        return LM_STUDIO_URL[: -len("/v1")]
    return LM_STUDIO_URL


def lm_loaded_context_length(model: str | None = None) -> int:
    """いまロードされているモデルの文脈長（トークン）を返す。分からなければ 0。

    LM Studio の REST API（``/api/v0/models``）だけが ``loaded_context_length`` を返す
    （OpenAI 互換の ``/v1`` には無い）。ここが分かって初めて「思考＋本文ぶんを残す」
    予算計算ができる。取得できないときは 0 を返し、呼び出し側は予算制御を行わず
    従来動作のままにする（RAG と同じく「無くても動く」層に留める）。
    """
    if LM_CONTEXT_LENGTH > 0:
        return LM_CONTEXT_LENGTH
    key = str(model or DEFAULT_MODEL)
    now = time.time()
    cached = _lm_context_cache.get(key)
    if cached and now - cached[0] < LM_CONTEXT_PROBE_TTL:
        return cached[1]
    value = 0
    try:
        base = _lm_server_root()
        with urllib.request.urlopen(f"{base}/api/v0/models", timeout=5) as res:
            entries = json.loads(res.read().decode("utf-8")).get("data") or []
        loaded = [
            item
            for item in entries
            if str(item.get("state") or "") == "loaded"
            and int(item.get("loaded_context_length") or 0) > 0
        ]
        for item in loaded:
            if _model_id_matches(key, str(item.get("id") or "")):
                value = int(item["loaded_context_length"])
                break
        # 名前が一致しなくても（リポジトリ+ファイル名指定など）ロード済みが分かれば
        # そちらが使われる。複数ロード時は最小値を採って安全側へ。
        if not value and loaded:
            value = min(int(item["loaded_context_length"]) for item in loaded)
    except Exception as exc:
        print(f"[ctx] loaded context probe failed: {type(exc).__name__}: {exc}")
        value = 0
    _lm_context_cache[key] = (now, value)
    if value:
        print(f"[ctx] loaded context length = {value} tok (model={key})")
    return value


class LMContextOverflowError(RuntimeError):
    """プロンプト自体が文脈長を超えて LM Studio に 400 で弾かれたときの例外。

    構造化出力の非対応 400 と区別するために独立した型にしている（混ぜると
    ``_lm_structured_output_unsupported`` を誤って立ててしまう）。
    """


def _read_http_error(err: urllib.error.HTTPError) -> str:
    with contextlib.suppress(Exception):
        return err.read().decode("utf-8", "replace")
    return ""


def _looks_like_context_overflow(detail: str) -> bool:
    text = str(detail or "").lower()
    return any(
        mark in text
        for mark in ("exceed_context_size", "context size", "context length", "n_ctx")
    )


def _choice_content(data: dict) -> str:
    try:
        return str(data["choices"][0]["message"].get("content") or "")
    except Exception:
        return ""


def _lm_slot_ids() -> list[int] | None:
    """``GET /slots`` でスロット番号の一覧を取る。取れなければ None。

    llama-server は ``[{"id":0,...},...]`` を返す（ビルドによっては
    ``{"slots":[...]}``）。LM Studio の内蔵エンジンはこの API を持たないので
    None になり、呼び出し側は id=0 だけを当てに行く。
    """
    try:
        url = f"{_lm_server_root()}/slots"
        with urllib.request.urlopen(url, timeout=LM_KV_CACHE_RELEASE_TIMEOUT) as res:
            payload = json.loads(res.read().decode("utf-8"))
    except Exception:
        return None
    entries = payload if isinstance(payload, list) else None
    if entries is None and isinstance(payload, dict):
        entries = payload.get("slots")
    if not isinstance(entries, list):
        return None
    ids: list[int] = []
    for item in entries:
        if isinstance(item, dict) and item.get("id") is not None:
            with contextlib.suppress(Exception):
                ids.append(int(item["id"]))
    return ids or None


def _lm_erase_slot(slot_id: int) -> bool:
    """``POST /slots/{id}?action=erase`` でスロットの KV キャッシュを捨てさせる。

    llama.cpp の save/restore は ``--slot-save-path`` が要るが、erase は不要。
    使用中のスロットは 4xx で断られる（直列化していれば通常起きない）。
    """
    req = urllib.request.Request(
        f"{_lm_server_root()}/slots/{slot_id}?action=erase",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=LM_KV_CACHE_RELEASE_TIMEOUT) as res:
            res.read()
        return True
    except urllib.error.HTTPError as err:
        print(
            f"[lm] slot {slot_id} erase failed: http {err.code} "
            f"{compact_text(_read_http_error(err), 160)}"
        )
        return False
    except Exception as exc:
        print(f"[lm] slot {slot_id} erase failed: {type(exc).__name__}: {exc}")
        return False


def _torch_vram_usage() -> str:
    """自プロセスの torch が抱えている VRAM を返す。

    Windows(WDDM) の nvidia-smi はプロセス別の使用量を報告しない（GeForce では
    ``--query-compute-apps`` が空で返る）ため、切り分けの決め手はこちら。
    ``reserved`` が torch のキャッシュアロケータが握っている総量で、``allocated``
    は実際に使っているぶん。差分がそのまま「返せるはずの居残り」になる。
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return "self-torch=not-loaded"
    try:
        if not torch.cuda.is_available():
            return "self-torch=no-cuda"
        reserved = int(torch.cuda.memory_reserved() / 1024 / 1024)
        allocated = int(torch.cuda.memory_allocated() / 1024 / 1024)
        return f"self-torch reserved={reserved}MiB allocated={allocated}MiB"
    except Exception as exc:
        return f"self-torch=unavailable({type(exc).__name__})"


def gpu_memory_snapshot() -> str:
    """GPU の使用量を「全体＋自プロセスの torch ぶん」で 1 行にまとめる。

    誰が VRAM を抱えているかはこれを見ないと分からない。実測で「推論サーバの KV
    キャッシュを erase しても全体は減らない」ことが分かっているので、
    ``全体`` と ``self-torch reserved`` の両方を並べて次のように読む。
      ・掃除で self-torch reserved が減り全体も減る → 犯人は TTS(このプロセス)
      ・self-torch reserved が小さいのに全体が大きい → 犯人は推論サーバ側。
        erase では戻らないので、サーバの並列数/文脈長など起動設定で削るしかない
    nvidia-smi が無い環境（AMD/Apple/CPU 実行）では 1 度試して以後全体値を諦める。
    """
    global _vram_probe_unsupported
    if not VRAM_MEMORY_LOG:
        return ""
    mine = _torch_vram_usage()
    if _vram_probe_unsupported:
        return mine
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=True,
        )
        line = proc.stdout.strip().splitlines()[0]
        used_mb, total_mb = (int(float(v.strip())) for v in line.split(",")[:2])
    except Exception as exc:
        _vram_probe_unsupported = True
        print(f"[vram] nvidia-smi probe unavailable: {type(exc).__name__}: {exc}")
        return mine
    return f"total={used_mb}/{total_mb}MiB  {mine}"


def release_torch_cuda_cache() -> int:
    """自プロセスの torch が抱えた CUDA キャッシュをドライバへ返す。返した MiB を返す。

    Irodori-TTS はこのプロセス内の GPU で走る（IRODORI_MODEL_DEVICE=auto なら cuda）。
    torch のキャッシュアロケータは解放済みブロックを再利用のため抱え続けるので、
    合成した文の長さぶんだけ VRAM が増えて戻らない。empty_cache() で明示的に返す。
    torch が未ロード／CUDA 無しなら何もしない（TTS をリモートに出す構成でも安全）。
    合成の真っ最中に呼ぶと、これから使い回すブロックまで返してしまうので、
    Irodori_lock が空いているときだけ実行する。
    """
    if not VRAM_RELEASE_TORCH:
        return 0
    torch = sys.modules.get("torch")
    if torch is None:  # まだ TTS を一度も動かしていない（＝抱えていない）
        return 0
    if not Irodori_lock.acquire(blocking=False):
        return 0
    try:
        if not torch.cuda.is_available():
            return 0
        before = int(torch.cuda.memory_reserved() / 1024 / 1024)
        # 参照が切れただけのテンソルを先に回収させないと、empty_cache は何も返せない。
        gc.collect()
        torch.cuda.empty_cache()
        with contextlib.suppress(Exception):
            torch.cuda.ipc_collect()
        after = int(torch.cuda.memory_reserved() / 1024 / 1024)
        freed = max(0, before - after)
        if freed:
            print(f"[vram] torch cache released: {freed}MiB (reserved {before} -> {after}MiB)")
        return freed
    except Exception as exc:
        print(f"[vram] torch cache release failed: {type(exc).__name__}: {exc}")
        return 0
    finally:
        Irodori_lock.release()


def release_lm_kv_cache(reason: str = "") -> bool:
    """推論サーバが抱えたプロンプト KV キャッシュを明示的に解放させる。

    ここまで来る時点で進行中のリクエストは無い（直列化＋在庫数 0 で呼ばれる）ので、
    全スロットを erase して VRAM を会話前の水準へ巻き戻す。1 つも解放できなければ
    「このサーバは非対応」と記録し、以後は毎ターン試さない（ログを汚さないため）。
    例外は投げない。戻り値は 1 つ以上解放できたか。
    """
    global _lm_kv_release_unsupported
    if LM_KV_CACHE_RELEASE == "off" or _lm_kv_release_unsupported:
        return False
    # 一覧が取れないビルド（`--slots` 無効など）でも erase だけは通ることがあるので、
    # 既定スロット 0 を当てに行ってから諦める。
    slot_ids = _lm_slot_ids() or [0]
    erased = sum(1 for slot_id in slot_ids if _lm_erase_slot(slot_id))
    if not erased:
        _lm_kv_release_unsupported = True
        print(
            "[lm] KV cache release unavailable on this server -> give up "
            "(enable the /slots endpoint on llama-server, or set LM_KV_CACHE_RELEASE=off)"
        )
        return False
    print(f"[lm] KV cache released: {erased} slot(s)" + (f" ({reason})" if reason else ""))
    return True


def release_vram(reason: str = "") -> None:
    """ターン終わりの掃除。自プロセスの torch キャッシュと推論サーバの KV を解放する。

    前後で nvidia-smi のプロセス別内訳をログへ出す。減らないときにどちらのプロセスが
    抱えているかを、これ 1 行で切り分けられるようにするため（推論サーバ側の KV バッファは
    n_ctx ぶん確保しきりなので erase では減らない＝「other が減らない」なら対策は
    サーバの起動設定側にある、という読み方をする）。
    """
    before = gpu_memory_snapshot()
    if before:
        print(f"[vram] before release ({reason or 'idle'}): {before}")
    release_torch_cuda_cache()
    release_lm_kv_cache(reason)
    after = gpu_memory_snapshot()
    if after:
        print(f"[vram] after  release ({reason or 'idle'}): {after}")


def _vram_sweep_enabled() -> bool:
    """ターン終わりの掃除を予約する意味があるか（両方 off なら予約しない）。"""
    return LM_KV_CACHE_RELEASE == "idle" or VRAM_RELEASE_TORCH


def _cancel_vram_release() -> None:
    """予約済みの遅延解放を取り消す（新しいリクエストが始まったとき）。"""
    global _lm_release_timer
    with _lm_release_lock:
        if _lm_release_timer is not None:
            _lm_release_timer.cancel()
            _lm_release_timer = None


def _schedule_vram_release() -> None:
    """LM_KV_CACHE_RELEASE_DELAY 秒後に 1 回だけ解放するよう予約し直す。"""
    global _lm_release_timer
    with _lm_release_lock:
        if _lm_release_timer is not None:
            _lm_release_timer.cancel()
        timer = threading.Timer(max(0.0, LM_KV_CACHE_RELEASE_DELAY), _deferred_vram_release)
        timer.daemon = True
        _lm_release_timer = timer
        timer.start()


def _deferred_vram_release() -> None:
    """静まったあとに 1 度だけ VRAM を掃除する（タイマースレッドから呼ばれる）。

    予約は LM リクエストと TTS 合成の両方の終わりから張られる。どちらが最後の GPU 仕事
    だったかに関係なく「静まってから 1 回」にしたいので、予約は 1 本だけ持ち回す。
    掃除中に次のリクエストが割り込むと「使用中スロット」を erase しようとして断られる
    ので、直列化ロックを取ってから叩く。取れなければ既に次が走っているということなので
    何もしない（そのリクエストの終了時に改めて予約される）。
    """
    global _lm_release_timer
    with _lm_release_lock:
        _lm_release_timer = None
    if not LM_SERIALIZE_REQUESTS:
        with _lm_inflight_lock:
            if _lm_inflight > 0:
                return
        release_vram("idle")
        return
    if not _lm_request_lock.acquire(timeout=0.2):
        return
    try:
        with _lm_inflight_lock:
            if _lm_inflight > 0:
                return
        release_vram("idle")
    finally:
        _lm_request_lock.release()


@contextlib.contextmanager
def _lm_request_slot():
    """LM Studio への 1 リクエストを囲み、直列化と終了後の KV 解放を担う。

    直列化の狙いは速度ではなく VRAM。サーバは同時リクエストのぶんだけスロット
    （＝別の KV キャッシュ）を確保し、空いても手放さない。返答本文の生成中に
    バックグラウンドの事実抽出が飛び込むと 2 本目が居残る（実測 +1.6GB）。
    each は即時解放、idle はタイマーで一拍置いてから解放する（返答本文と補助生成の
    間には TTS 合成ぶんの空白があり、即時だと 1 ターンで何度も掃除してしまう）。
    """
    global _lm_inflight
    _cancel_vram_release()
    if LM_SERIALIZE_REQUESTS:
        _lm_request_lock.acquire()
    with _lm_inflight_lock:
        _lm_inflight += 1
    try:
        yield
    finally:
        with _lm_inflight_lock:
            _lm_inflight -= 1
            idle = _lm_inflight <= 0
        try:
            if LM_KV_CACHE_RELEASE == "each":
                release_vram("each")
        finally:
            if LM_SERIALIZE_REQUESTS:
                _lm_request_lock.release()
        if idle and _vram_sweep_enabled():
            _schedule_vram_release()


def _disable_prompt_cache(aux: bool) -> bool:
    """このリクエストに cache_prompt=false を付けるか判定する。

    ``aux`` は返答本文ではない補助生成（クエリ書き換え・事実抽出・要点メモ）。
    既定 ``aux`` では補助生成だけキャッシュを無効化する。補助生成のプロンプトは
    毎回中身が違って再利用が効かないうえ、スロットに載ると返答本文が再利用したい
    履歴の接頭辞を追い出してしまうため、残す価値がそもそも無い。
    """
    if LM_CACHE_PROMPT == "off" or _lm_cache_prompt_unsupported:
        return False
    if LM_CACHE_PROMPT == "all":
        return True
    return aux


def _use_structured_output(segmented_mode: bool) -> bool:
    """このリクエストで response_format(json_schema) を付けるか判定する。

    構造化出力は segments スキーマを強制するため、segments を期待する segmented_mode の
    ときだけ有効。auto では過去にサーバが拒否していなければ付ける。
    """
    if not segmented_mode or LM_STRUCTURED_OUTPUT == "off":
        return False
    if LM_STRUCTURED_OUTPUT == "on":
        return True
    return not _lm_structured_output_unsupported  # auto


def _post_lmstudio_chat(payload: dict, use_structured: bool, aux: bool = False) -> dict:
    """chat/completions に POST し、レスポンス JSON を返す。

    ``use_structured`` が True のときだけ response_format(json_schema) を付ける。
    ``aux`` が True（返答本文ではない補助生成）なら cache_prompt=false を付け、
    プロンプトの KV キャッシュをサーバに残させない（LM_CACHE_PROMPT で範囲を変更可）。
    どちらも非対応サーバがあり得るため、HTTP 4xx なら該当フィールドを外して 1 度だけ
    再送し、以後は付けないようプロセス内に記録する。どちらが原因かはエラー本文に出た
    フィールド名で切り分け、分からなければ拒否例の多い response_format から疑う。
    プロンプトが文脈長を超えた 400 はどちらとも無関係なので LMContextOverflowError と
    して区別する（混ぜると構造化出力やキャッシュ制御が以後ずっと外れてしまう）。
    リクエストは _lm_request_slot() で囲む（直列化＋捌け際の KV 解放）。
    """
    global _lm_structured_output_unsupported, _lm_cache_prompt_unsupported

    last_detail = ""

    def _send(body: dict) -> dict:
        nonlocal last_detail
        req = urllib.request.Request(
            f"{LM_STUDIO_URL}/chat/completions",
            data=json_bytes(body),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_detail = ""
        with _lm_request_slot():
            try:
                with urllib.request.urlopen(req, timeout=LM_STUDIO_TIMEOUT) as res:
                    return json.loads(res.read().decode("utf-8"))
            except urllib.error.HTTPError as err:
                last_detail = _read_http_error(err)
                print(f"[lm] http {err.code}: {compact_text(last_detail, 300)}")
                if err.code == 400 and _looks_like_context_overflow(last_detail):
                    raise LMContextOverflowError(compact_text(last_detail, 300)) from err
                raise

    no_cache = _disable_prompt_cache(aux)
    body = dict(payload)
    if use_structured:
        body["response_format"] = LM_SEGMENT_RESPONSE_FORMAT
    if no_cache:
        # llama.cpp 系の拡張フィールド。OpenAI 互換の /v1 でもそのまま受け取る。
        body["cache_prompt"] = False
    if not use_structured and not no_cache:
        return _send(body)
    try:
        return _send(body)
    except urllib.error.HTTPError as err:
        if not (400 <= err.code < 500):
            raise
        detail = str(last_detail).lower()
        retry = dict(body)
        if no_cache and "cache_prompt" in detail:
            _lm_cache_prompt_unsupported = True
            print("[lm] server rejected cache_prompt -> stop sending it")
            retry.pop("cache_prompt", None)
            return _send(retry)
        if use_structured:
            _lm_structured_output_unsupported = True
            retry.pop("response_format", None)
            return _send(retry)
        _lm_cache_prompt_unsupported = True
        print("[lm] server rejected the request body -> stop sending cache_prompt")
        retry.pop("cache_prompt", None)
        return _send(retry)


# RAG 想起クエリの LLM 書き換え設定（環境変数で調整可）。
_QUERY_REWRITE_ENABLED = os.environ.get("RAG_QUERY_REWRITE", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
    "",
}
_QUERY_REWRITE_MAXTOK = int(os.environ.get("RAG_QUERY_REWRITE_MAXTOK", "64"))
# 書き換えLLM呼び出しの生成モード。空ならチャットと同じ generation_mode に追従する。
# 例: 品質最優先(unlimited)でチャットしつつ、書き換えだけ prefill で高速化したいとき指定。
_QUERY_REWRITE_MODE = os.environ.get("RAG_QUERY_REWRITE_MODE", "").strip().lower()
# 想起クエリを最大何本生成するか。「全部挙げて」等の列挙・網羅質問は 1 本のクエリだと
# 全列挙が top-k に埋もれるため、観点違いのクエリを複数本出して和集合で網を広げる。
# 1 なら従来どおり単一クエリ。和集合は同一記憶を最高スコアで重複除去してから top_k で切る。
_QUERY_REWRITE_MULTI = max(1, int(os.environ.get("RAG_QUERY_REWRITE_MULTI", "3")))
# 想起の件数/閾値の独立ノブ（tools/diagnose_recall.py で実測してから調整する用）。
# top_k=16: 実測で e5 のスコアは 0.78〜0.88 の狭帯に圧縮され、料理系の記憶が団子状態で
# 30 件すべて min_score(0.75) 以上だった。つまり閾値ではなく件数が律速で、枠が狭いと列挙
# 質問の実料理（塩じゃけ/シソ餃子/かつおのたたき等）が圏外へこぼれる。全件が高スコアなので
# 枠を広げても品質は落ちない（近重複は _recall_dup_signature で畳んでから切るため水増しも
# 起きない）。min_score は下げても無意味（閾値未満の除外料理は存在しない）ため 0.75 据え置き。
# いずれも RAG_RECALL_TOP_K / RAG_RECALL_MIN_SCORE で上書き可。
_RECALL_TOP_K = int(os.environ.get("RAG_RECALL_TOP_K", "16"))
_RECALL_MIN_SCORE = float(os.environ.get("RAG_RECALL_MIN_SCORE", "0.75"))
# 近重複の集約を切る用（RAG_RECALL_DEDUP=0 で無効化）。集約が別料理まで畳んで
# いないか等を、再コンパイル無しで A/B するための逃がし弁。
_RECALL_DEDUP = os.environ.get("RAG_RECALL_DEDUP", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
    "",
}


# --- 時系列・列挙の意図検出 ---------------------------------------------------
# 時系列想起の候補プール。時系列で選ぶ前に確保する広さで、狭いと最古/最新が
# cosine 上位枠から溢れて取りこぼす（スコアは狭帯に潰れており順位は当てにならない）。
_RECALL_TEMPORAL_POOL_K = int(os.environ.get("RAG_TEMPORAL_POOL_K", "64"))
# 時系列質問で最終的にプロンプトへ載せる件数（年表の行数）。
_RECALL_TEMPORAL_K = int(os.environ.get("RAG_TEMPORAL_K", "8"))
# 時系列選抜で「話題の芯」と見なすスコア帯（最高スコアからの許容差）。
# これが無いと、閾値ぎりぎりの無関係な古い記憶が「一番最初」の座を奪う。
# 0.02 は実測にもとづく値: e5-small のスコアは 0.80〜0.84 に潰れており、
# 「本を買った」系の記録が 0.828〜0.841、無関係な「はじめまして」が 0.801 だった。
# 0.04 だと挨拶まで帯に入り、「一番最初に買ってあげた本」の答えが挨拶にすり替わる。
_RECALL_TEMPORAL_BAND = float(os.environ.get("RAG_TEMPORAL_BAND", "0.02"))
# 語彙チャネルから載せる上限。列挙質問では網羅性を優先して広げる。
_RECALL_LEXICAL_LIMIT = int(os.environ.get("RAG_LEXICAL_LIMIT", "24"))
_RECALL_LEXICAL_LIMIT_ENUM = int(os.environ.get("RAG_LEXICAL_LIMIT_ENUM", "48"))
# 語彙一致は独立した証拠なので、ベクトルの閾値より少し緩めて採用する。
_RECALL_LEXICAL_SLACK = float(os.environ.get("RAG_LEXICAL_SLACK", "0.03"))
# 台帳から載せる事実の上限（列挙は集計なので既定を広く取る）。
_RECALL_LEDGER_LIMIT = int(os.environ.get("RAG_LEDGER_LIMIT", "60"))
# 台帳に載せた事実の出典として、原文の往復を何件まで年表へ含めるか
# （一覧と年表が食い違わないための裏付け。多すぎると文脈を圧迫する）。
_RECALL_LEDGER_TURNS = int(os.environ.get("RAG_LEDGER_TURNS", "8"))
# 「した事」を訊かれたときに、予定・願望を**打ち消し材料として**何件まで併記するか。
# 黙って落とすと、年表に残る原文（「次は〜作ってあげるね」）だけを見た LLM が
# 「作った料理」として挙げてしまう。0 で併記しない。
_RECALL_LEDGER_PLANS = int(os.environ.get("RAG_LEDGER_PLANS", "12"))
# 事実抽出 LLM の生成上限（短い JSON を返させるだけなので小さく保つ）。
_FACT_EXTRACT_MAXTOK = int(os.environ.get("RAG_FACT_EXTRACT_MAXTOK", "256"))
# 質問が時系列・列挙かどうかに関わらず、台帳を常に併用するか（0 なら時系列/列挙時のみ）。
_LEDGER_ALWAYS = os.environ.get("RAG_LEDGER_ALWAYS", "0").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
    "",
}
# 毎ターンの増分抽出（返答後に今回の往復を台帳へ）。LLM 抽出を含むので必ず別スレッド。
_LEDGER_LIVE = os.environ.get("RAG_LEDGER_LIVE", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
    "",
}
# 増分抽出で LLM を使うか（0 ならルール抽出のみ＝完全にゼロコスト）。
_LEDGER_LIVE_LLM = os.environ.get("RAG_LEDGER_LIVE_LLM", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
    "",
}

# 質問形の目印。時系列の並べ替えは「問われたとき」だけ効かせたいので、
# 平叙文（「最初は苦手だったけど」等）で誤発火しないよう質問形を要求する。
_QUESTION_RE = re.compile(
    r"[?？]|だっけ|だったっけ|かな[?？]?$|覚えてる|おぼえてる|教えて|なんだ|何だ|"
    r"なに|何|どれ|どっち|いつ|言って|挙げて|あげて$"
)
# 「一番最初」系（最古を答えさせる）。
_FIRST_RE = re.compile(
    r"(?:一番|いちばん|1番|最も|もっとも)\s*(?:最初|はじめ|初め|古い)|"
    r"最初に|最初の|初めて|はじめて|初の|初回|最古|"
    r"出会った(?:頃|ころ|とき|時|ばかり)|知り合った(?:頃|ころ|とき|時)|"
    r"付き合い始め|一番古い"
)
# 「一番最後・最近」系（最新を答えさせる）。
_LAST_RE = re.compile(
    r"(?:一番|いちばん|1番)\s*(?:最後|新しい|最近)|最後に|最後の|最新|直近|"
    r"この前|前回|さっき|最近"
)
# 「いつ？」系（日付・経過期間を答えさせる）。
_WHEN_RE = re.compile(
    r"いつ|何年|何月|何日|何ヶ月|何か月|どのくらい前|どれくらい前|どれぐらい前|時期|"
    r"何年前|何日前"
)
# 列挙・網羅を求める発話（台帳と語彙チャネルの網羅性を効かせる）。
_ENUM_RE = re.compile(
    r"全部|ぜんぶ|すべて|全て|他に|ほかに|他の|いくつ|何個|何品|一覧|"
    r"思いつく|挙げて|あげて|列挙|残らず|漏れなく"
)


def _resolve_period(text: str, today: date) -> tuple[str, str, str]:
    """発話に含まれる期間表現を (since, until, ラベル) へ解決する。

    実装は fact_extract 側にある（抽出でも同じ時間表現を「出来事の時期」へ解決するので、
    2 か所に同じ規則を持つと必ず食い違う）。fact_extract が読めない環境では期間で
    絞らないだけで、想起そのものは従来どおり動く。
    """
    if fact_extract is None:
        return "", "", ""
    try:
        return fact_extract.resolve_period(str(text or ""), today)
    except Exception:
        return "", "", ""


def detect_recall_intent(user_text: str, today: date | None = None) -> dict:
    """発話から想起の意図（時系列の向き・列挙・期間）を検出する。

    戻り値:
      ``temporal``  'first'（最古を答える） / 'last'（最新） / 'when'（日付を答える） / ''
      ``enum``      列挙・網羅を求めているか
      ``since`` / ``until`` / ``period``  期間の絞り込みとその表示名

    時系列の並べ替えは質問のときだけ効かせる（平叙文で誤発火させない）。判定を外しても
    壊れないのが前提の設計で、外れた場合は従来どおりスコア順の想起になるだけ。
    """
    text = str(user_text or "").strip()
    # question は台帳の絞り込み（主客・動詞・カテゴリの推定）に使うので原文を持ち回る。
    result = {
        "temporal": "",
        "enum": False,
        "since": "",
        "until": "",
        "period": "",
        "question": text,
    }
    if not text:
        return result
    result["enum"] = bool(_ENUM_RE.search(text))
    since, until, period = _resolve_period(text, today or date.today())
    result["since"] = since
    result["until"] = until
    result["period"] = period
    if not _QUESTION_RE.search(text):
        return result
    # 「一番最初」→「最後」→「いつ」の順に見る（「最初に会ったのはいつ？」は first 優先。
    # 最古を答えるのが主目的で、日付はそこに添えればよい）。
    if _FIRST_RE.search(text):
        result["temporal"] = "first"
    elif _LAST_RE.search(text):
        result["temporal"] = "last"
    elif _WHEN_RE.search(text):
        result["temporal"] = "when"
    return result


def _recall_dup_signature(mem: dict) -> str:
    """近重複を畳むための集約キー。同一シーンの記録（同じプロンプトが再生成・繰り返し
    プレイで別 ts に複数）を 1 つにまとめるため、user_text（無ければ reply_text）から
    空白・記号を除いた先頭 32 文字を署名にする。異なる料理は先頭が変わるので畳まれない
    （例:「今日は米のご飯と塩じゃけ…」の重複だけを集約し、麻辣麻婆とゴーヤ麻婆は別扱い）。"""
    text = str(mem.get("user_text") or "").strip() or str(mem.get("reply_text") or "")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[、。！？!?…・,.\"'`「」『』（）()]", "", text)
    return text[:32] or str(mem.get("ts") or "")


def rewrite_recall_queries(
    user_text: str,
    recent_context: str = "",
    model: str | None = None,
    generation_mode: str = "",
    user_name: str = "",
    char_name: str = "",
) -> tuple[list[str], str]:
    """ユーザー発話から、RAG 想起用の検索クエリを LLM で生成する（最大 N 本）。

    戻り値は ``(クエリ列, 意図タグ)``。意図タグは 'first' / 'last' / 'when' / 'enum' を
    カンマ区切りで含む文字列（不明なら空）。時系列・列挙の判定は正規表現
    （detect_recall_intent）を主とし、こちらは言い回しの取りこぼしを補うだけの補助。

    生の会話発話は挨拶・相槌・依頼の枕詞などの雑音が多く、意味検索（e5）の精度が落ちる
    （例:「それじゃ早速声を聞かせて…質問です…讃岐うどん以外に作った料理を挙げて」）。
    さらに除外・列挙質問には固有の弱点が 2 つある:
      ・除外語（「讃岐うどん以外」）をクエリに残すと、意味検索は否定を表現できず逆に
        その語へ検索が引きずられ、狙いの記憶（他の料理）の順位が下がる。
      ・「全部挙げて」は全列挙を求めるが、意味検索は類似度 top-k しか返せず、1 本の
        クエリでは網羅に構造的に弱い。
    そこで話題の核だけへ書き換え（除外語・否定語は捨てる）つつ、列挙質問では観点違いの
    クエリを複数本返す。呼び出し側は各クエリで recall して和集合を取る。失敗・空・無効化
    時は空リストを返し、呼び出し側は原文へフォールバックする（RAG は純粋な追加レイヤー）。
    """
    user_text = str(user_text or "").strip()
    if not _QUERY_REWRITE_ENABLED or not user_text:
        return [], ""
    n = _QUERY_REWRITE_MULTI
    # 主体・客体は発話ごとに変わる（「ナデシコが破壊した」「オサムが作った」等）ので、
    # 上の汎用ルールで明示された主体・客体はそのまま保つ。ただし主語が省略され文脈でも
    # 特定できない行為質問（例:「作った料理は？」）に限っては、この会話が基本ユーザー→
    # キャラの関係であることを手掛かりに、主体をユーザー名で補ってアンカーする（フォール
    # バック）。名前が総称（あなた/君 等）や未設定なら補わない。
    _generic_names = {"", "あなた", "きみ", "君", "お前", "おまえ", "きみたち"}
    _u = str(user_name or "").strip()
    _c = str(char_name or "").strip()
    _who = ""
    actor_rule = ""
    if _u and _u not in _generic_names:
        _who = f"ユーザー={_u}" + (
            f"／キャラクター={_c}" if (_c and _c not in _generic_names) else ""
        )
        _to = f"相手（{_c}）に" if (_c and _c not in _generic_names) else ""
        actor_rule = (
            f"・上記で主体が発話に明示されているならそれを優先する。主体が省略され文脈でも"
            f"特定できない行為・出来事の質問に限り、この会話は基本『{_u}（ユーザー）が{_to}"
            f"行う』関係なので、主体を {_u} と補ってクエリに含める"
            f"（例: 主語のない『作った料理は？』→『{_u} 作った 料理』）。\n"
        )
    system = (
        "あなたは検索クエリ生成器です。ユーザーの発話から、過去の会話ログを意味検索する"
        "ための短い日本語クエリを出力します。次の規則に従ってください。\n"
        "・挨拶・相槌・依頼の枕詞（例:『声を聞かせて』『質問です』『思いつくだけ挙げて』）は捨てる。\n"
        "・知りたい事柄の核（名詞・動詞・固有名詞）だけを、1 行あたり 2〜10 語ほど空白区切りで並べる。\n"
        "・『〜以外』『〜を除いて』などで除外された語や否定語は絶対にクエリへ入れない"
        "（意味検索は否定を表現できず、その語に検索が引きずられて逆効果になるため）。\n"
        "・『作った・破壊した・言った・渡した・行った』等の行為や出来事を尋ねる発話では、"
        "その行為の主体（誰が/何が）・客体（何を/誰に/何に対して）・動詞をできるだけクエリに"
        "残し、『誰が（何が）何にしたか』の関係を保つ。主体・客体が発話に明示されていれば、"
        "それをユーザー/キャラに勝手に置き換えず、そのまま使う"
        "（例:『あのときナデシコがグラビティブラストで破壊したのは何だっけ？』→"
        "『ナデシコ グラビティブラスト 破壊 対象』）。\n"
        f"・「全部」「他には」「いくつも」等、網羅・列挙を求める発話では、観点や言い換えを"
        f"変えたクエリを 1 行 1 本で最大 {n} 行まで出す（例: 料理なら 1 行目『作った 料理 献立』、"
        "2 行目『夕飯 おかず 手料理』、3 行目『和食 中華 手料理』のように角度を変える）。"
        "会話ログは日常の場面なので、抽象的な総称語（一覧・メニュー・レシピ・カテゴリ等）は"
        "使わず、日常の具体語（夕飯・晩御飯・おかず・和食・中華・煮物 等）や想定される品目・"
        "食材で角度を変えること。単純な質問なら 1 行でよい。\n"
        + actor_rule
        + "・『いつ』『一番最初』『初めて』『最後』『最近』『去年』『先月』『3月』等、"
        "時期・順序・日付を尋ねる語や期間の表現はクエリに入れない（時期の絞り込みと"
        "時系列の並べ替えは検索とは別の仕組みで行うため、クエリには話題の核だけを残す。"
        "例:『一番最初に買ってあげた本は何だっけ？』→『買った 本 プレゼント』）。\n"
        "・1 行目に、その発話が何を求めているかのタグを `#intent:` として出す。"
        "使える値は first（最初・初めてを聞いている）/ last（最後・最近）/ when（いつ・時期）/ "
        "enum（全部・網羅・列挙）/ none（いずれでもない）で、複数該当ならカンマ区切り"
        "（例: `#intent: first,when`）。2 行目以降にクエリ本文を書く。\n"
        + "・説明・引用符・記号・箇条書き番号・思考は出さず、タグ行とクエリ本文だけを行区切りで返す。/no_think"
    )
    parts: list[str] = []
    if str(recent_context or "").strip():
        parts.append(f"直近の文脈（背景・参考）:\n{str(recent_context).strip()}")
    if _who:
        parts.append(f"登場人物: {_who}")
    parts.append(f"ユーザー発話:\n{user_text}")
    parts.append("検索クエリ:")
    # 複数行を出させるぶん、prefill/original モードのフォールバック上限を少し広げる。
    rewrite_maxtok = _QUERY_REWRITE_MAXTOK if n <= 1 else max(_QUERY_REWRITE_MAXTOK, 32 * n)
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
        "temperature": 0.0,
        "max_tokens": rewrite_maxtok,
        "stream": False,
    }
    # 生成モードごとに content 取得の作法が異なる（思考ONの unlimited/quality_guard は
    # 思考を完走させて content を埋め、prefill は空白assistantで思考を抑止、original は
    # そのまま＝思考モデルでは空になり得る）。プリフィルを直に付けると unlimited 等の
    # 思考ONモードと競合するため、通常チャットと同じ _request_lmstudio_content に委譲して
    # 現在のモードへ追従する。RAG_QUERY_REWRITE_MODE で上書き可。
    rewrite_mode = _QUERY_REWRITE_MODE or str(generation_mode or "").strip().lower()
    if rewrite_mode not in LM_GENERATION_MODES:
        rewrite_mode = DEFAULT_LM_GENERATION_MODE
    try:
        data = _request_lmstudio_content(
            payload,
            segmented_mode=False,
            auto_emoji=False,
            base_max_tokens=rewrite_maxtok,
            mode=rewrite_mode,
            # 数行のクエリを出すだけなので、思考ONモードでも枠は rewrite_maxtok で頭打ちにする。
            cap_to_base_max_tokens=True,
        )
        content = str(data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        return [], ""
    if not content:
        return [], ""
    # 各行を独立クエリとして拾い、「検索クエリ:」等のラベル・箇条書き記号・囲みの引用符を
    # 剥がしてから重複を除く（大小無視）。上限 n 本で打ち切る。
    # `#intent:` 行はクエリではなく意図タグとして取り分ける（出さないモデルもあるので、
    # 無い場合も正常系として扱い、呼び出し側は正規表現の判定だけで動く）。
    queries: list[str] = []
    seen: set[str] = set()
    intent = ""
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        intent_match = re.match(r"^#?\s*intent\s*[:：]\s*(.+)$", line, flags=re.IGNORECASE)
        if intent_match:
            tags = re.split(r"[,、\s/]+", intent_match.group(1).strip().lower())
            intent = ",".join(
                tag for tag in tags if tag in {"first", "last", "when", "enum"}
            )
            continue
        line = re.sub(r"^(検索クエリ|クエリ|query)\s*[:：]\s*", "", line, flags=re.IGNORECASE).strip()
        line = re.sub(r"^(?:[-*・>]+|\d+[.)、])\s*", "", line).strip()
        line = line.strip("「」『』\"'`  ").strip()
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(line)
        if len(queries) >= n:
            break
    return queries, intent


def _merge_recalled(pool: dict, memories: list[dict]) -> None:
    """想起結果を ts（無ければ本文）キーの辞書へ和集合で畳み込む（最高スコアを採用）。

    同じ往復が複数チャネル・複数クエリから来るので、ここで 1 件へまとめる。
    語彙チャネル由来（score=0 になり得る）が、ベクトル由来の高スコアを上書きしない
    ように、スコアは常に大きい方を残し、``via`` は両方を記録する。
    """
    for mem in memories:
        key = str(mem.get("ts") or "") or (
            f"{mem.get('user_text')}\n{mem.get('reply_text')}"
        )
        current = pool.get(key)
        if current is None:
            pool[key] = dict(mem)
            continue
        if float(mem.get("score") or 0) > float(current.get("score") or 0):
            merged_via = current.get("via")
            current.update(mem)
            if merged_via and merged_via != mem.get("via"):
                current["via"] = f"{merged_via}+{mem.get('via')}"
        elif mem.get("via") and mem.get("via") != current.get("via"):
            current["via"] = f"{current.get('via')}+{mem.get('via')}"


def _select_recalled(pool: list[dict], intent: dict) -> list[dict]:
    """想起プールから、意図に応じて最終的に載せる記憶を選ぶ。

    時系列質問（first/last）では、スコア順に切ってから並べ替えるのでは最古/最新を
    取りこぼす。逆に、緩い閾値のまま時系列で切ると無関係な古い記憶が「最初」の座を
    奪う。そこで「話題の芯（最高スコアから _RECALL_TEMPORAL_BAND 以内）」に絞って
    から時系列で選び、最後にスコア上位も少数だけ足して話題の軸を保つ。
    """
    if not pool:
        return []
    by_score = sorted(pool, key=lambda mem: -float(mem.get("score") or 0))
    temporal = str(intent.get("temporal") or "")
    if temporal not in {"first", "last"}:
        # when / 期間 / 通常はスコア順（提示側で年表に整形するかどうかが変わるだけ）。
        limit = _RECALL_TEMPORAL_K if temporal == "when" else _RECALL_TOP_K
        return by_score[:limit]
    band = _RECALL_TEMPORAL_BAND
    # by_score とは別リストにする（下の in-place ソートで by_score の順序が壊れると、
    # 後段の「スコア上位も少数だけ含める」が実際には最古 2 件を足してしまう）。
    focused = list(by_score)
    if band > 0 and by_score:
        floor = float(by_score[0].get("score") or 0) - band
        # 語彙チャネル由来でスコアが付いていない行（score=0）は帯で落とさない。
        # 語の一致という独立した証拠があるため、cosine の帯だけで切ると
        # 「最古の 1 件」を語彙チャネルで拾った意味が無くなる。
        focused = [
            mem
            for mem in by_score
            if float(mem.get("score") or 0) >= floor or not mem.get("score")
        ] or list(by_score)
    # 主題の直接証拠がある記憶（検索語が実際に本文へ一致した／台帳の出典である）を優先する。
    # cosine のスコア帯だけには頼れない: e5-small では無関係な挨拶(0.801)と本の記録(0.841)の
    # 差が帯の幅と同程度しかなく、帯だけで絞ると「一番最初」が挨拶にすり替わる（実測）。
    # 語の一致は主題の直接証拠なので、こちらがあれば時系列選抜はその中だけで行う。
    evidenced = [
        mem
        for mem in focused
        if "keyword" in str(mem.get("via") or "") or "ledger" in str(mem.get("via") or "")
    ]
    if evidenced:
        focused = evidenced
    focused.sort(
        key=lambda mem: rag_memory.ts_sort_key(mem.get("ts")),
        reverse=(temporal == "last"),
    )
    picked = focused[: _RECALL_TEMPORAL_K]
    picked_keys = {id(mem) for mem in picked}
    # 話題の軸を保つため、スコア最上位も 2 件までは必ず含める（年表なので
    # 追加しても時系列の位置に並ぶだけで「先頭＝最古」は崩れない）。
    for mem in by_score[:2]:
        if id(mem) not in picked_keys:
            picked.append(mem)
    return picked


def recall_for_turn(
    character_id: str,
    *,
    queries: list[str],
    slot: str,
    mode: str,
    intent: dict,
    recent_user_texts: list,
    user_name: str = "",
    char_name: str = "",
) -> dict:
    """ベクトル・語彙・台帳の 3 チャネルで想起し、プロンプト用のブロックを組む。

    どのチャネルも「あれば使う」追加レイヤーとして扱い、失敗・0 件でも他が動く。
    戻り値は ``{"block": str, "mode": str, "stats": dict}``。``mode`` は
    request_lmstudio へ渡す提示形式（''／'timeline'／'ledger'／'timeline+ledger'）。
    """
    result = {"block": "", "mode": "", "stats": {}}
    if rag_memory is None:
        return result
    temporal = str(intent.get("temporal") or "")
    enum = bool(intent.get("enum"))
    since = str(intent.get("since") or "")
    until = str(intent.get("until") or "")
    order = {"first": "oldest", "last": "newest"}.get(temporal, "score")
    pool: dict = {}
    # 台帳を引くための絞り込み条件（主体・行為・向き・カテゴリ）を先に推定する。
    # 行為を尋ねる質問（verb が取れた質問）は、時系列・列挙でなくても台帳を使う:
    # 「君が俺に作ってくれた料理は？」は意味検索だと「俺が君に作ってあげた」記憶と
    # ほぼ同じベクトルになり、主客を取り違えた回答（「私は作っていません」等）の
    # 原因になる。向きを構造で持つ台帳はここでこそ効く。
    filters: dict = {}
    if fact_extract is not None:
        filters = fact_extract.infer_query_filters(
            " ".join([str(intent.get("question") or "")] + list(queries)),
            user_name=user_name,
            char_name=char_name,
        )
    # 台帳は「絞り込める」ときだけ使う。動詞・カテゴリ・向きのどれも決まらない質問で
    # 引くと、質問と無関係な最古の事実から順に上限件数ぶん並べるだけになり
    # （実測: 「マリンタワーに登ったのはいつ？」で無関係な60件が混入した）、
    # 出典として年表へ持ち込む原文まで無関係なもので埋まる。
    has_ledger_filter = any(
        str(filters.get(key) or "").strip() for key in ("verb", "category", "direction")
    )
    use_ledger = has_ledger_filter and bool(
        temporal or enum or since or _LEDGER_ALWAYS or filters.get("verb")
    )

    # 1) ベクトル（意味）チャネル: 従来の想起。時系列質問ではプールを広げて ts で並べ替える。
    for query in queries:
        try:
            hits = rag_memory.recall_memory(
                character_id,
                query,
                k=_RECALL_TEMPORAL_K if order != "score" else _RECALL_TOP_K,
                slot=slot,
                mode=mode,
                recent_user_texts=recent_user_texts,
                min_score=_RECALL_MIN_SCORE,
                order=order,
                pool_k=_RECALL_TEMPORAL_POOL_K,
                since=since,
                until=until,
                score_band=_RECALL_TEMPORAL_BAND if order != "score" else 0.0,
            )
        except Exception:
            hits = []
        _merge_recalled(pool, hits)
    result["stats"]["vector"] = len(pool)

    # 2) 語彙チャネル: 件数上限のない一致検索。cosine top-k が構造的に取りこぼす
    #    「最古の 1 件」「該当の全件」をここで補う。
    lexical_count = 0
    if temporal or enum or since or use_ledger:
        keywords: list[str] = []
        for query in queries:
            keywords.extend(rag_memory.normalize_keywords(query))
        try:
            lexical = rag_memory.search_lexical(
                character_id,
                keywords,
                slot=slot,
                mode=mode,
                since=since,
                until=until,
                order={"last": "newest"}.get(temporal, "oldest"),
                limit=_RECALL_LEXICAL_LIMIT_ENUM if enum else _RECALL_LEXICAL_LIMIT,
                rescore_query=queries[0] if queries else "",
                min_score=max(0.0, _RECALL_MIN_SCORE - _RECALL_LEXICAL_SLACK),
                # 列挙質問では網羅性を優先して 1 語一致でも拾う。既定（3 語以上なら 2 語
                # 一致を要求）だと、「作った 料理 献立」に対して「肉じゃがを作ってあげた」が
                # 『作』しか一致せず落ちる＝列挙の主役が全部消える（実測）。
                # 誤爆は再スコアと台帳・原文の併記で吸収する。
                min_hits=1 if enum else 0,
            )
        except Exception:
            lexical = []
        lexical_count = len(lexical)
        _merge_recalled(pool, lexical)
    result["stats"]["lexical"] = lexical_count

    # 3) 台帳チャネル: 列挙・時系列は検索ではなく集計。SQL で全件が返るので、
    #    主客（誰が誰に）を取り違えず、件数の窓による漏れも起きない。
    facts: list[dict] = []
    if use_ledger and fact_extract is not None:
        try:
            facts = rag_memory.query_facts(
                character_id,
                category=filters.get("category", ""),
                verb=filters.get("verb", ""),
                direction=filters.get("direction", ""),
                # その人自身の行為（direction='self'）も一緒に拾う。「俺が行った場所」は
                # 相手への行為ではないので、向きの一致だけでは丸ごと漏れる。
                self_subject=filters.get("self_subject", ""),
                # 相の絞り込み（既定 'done'）。これが無いと「今度作ってあげるね」と
                # 言っただけの料理が「作った料理」として列挙に混ざる。
                modality=filters.get("modality", ""),
                slot=slot,
                mode=mode,
                since=since,
                until=until,
                order="newest" if temporal == "last" else "oldest",
                limit=_RECALL_LEDGER_LIMIT,
            )
        except Exception:
            facts = []
    # 「した事」を訊かれたときは、同じ行為の**予定・願望も打ち消し材料として**添える。
    # 一覧から黙って落とすと、年表に残る原文（「次はうどんなカルボナーラを作ってあげるね」）
    # だけを読んだ LLM が「作った料理」として挙げてしまう。一覧では別の節に分かれ、
    # 「まだしていない」と明示されるので、混ざるのではなく打ち消しとして働く。
    plans: list[dict] = []
    if facts and _RECALL_LEDGER_PLANS > 0 and filters.get("modality") == "done":
        try:
            plans = rag_memory.query_facts(
                character_id,
                category=filters.get("category", ""),
                verb=filters.get("verb", ""),
                direction=filters.get("direction", ""),
                self_subject=filters.get("self_subject", ""),
                modality="plan",
                slot=slot,
                mode=mode,
                since=since,
                until=until,
                order="newest",
                limit=_RECALL_LEDGER_PLANS,
            )
        except Exception:
            plans = []
        facts = facts + plans
    result["stats"]["ledger"] = len(facts)
    result["stats"]["ledgerPlans"] = len(plans)
    result["stats"]["filters"] = filters

    # 台帳に載せる事実の出典（原文の往復）を必ず想起プールへ入れる。
    # 台帳は索引にすぎないので、一覧だけを見せると LLM は抽出ミスを検証できない。
    # さらに一覧と年表が別の出来事を指していると（例: 一覧は「星の王子さま」、年表は
    # 「写真集」）、どちらが最初なのか判断できず誤答の原因になる。
    ledger_sources = []
    for fact in facts:
        source_id = fact.get("source_id")
        if source_id and source_id not in ledger_sources:
            ledger_sources.append(source_id)
    ledger_turns: list[dict] = []
    if ledger_sources:
        try:
            ledger_turns = rag_memory.fetch_turns(
                character_id,
                ledger_sources[:_RECALL_LEDGER_TURNS],
                slot=slot,
                mode=mode,
            )
        except Exception:
            ledger_turns = []
        _merge_recalled(pool, ledger_turns)
    result["stats"]["ledgerTurns"] = len(ledger_turns)

    # 近重複の集約（同一シーンが別 ts で何度も記録されている分を 1 つに畳む）。
    merged = list(pool.values())
    if _RECALL_DEDUP:
        collapsed: dict = {}
        for mem in merged:
            signature = _recall_dup_signature(mem)
            current = collapsed.get(signature)
            if current is None or float(mem.get("score") or 0) > float(
                current.get("score") or 0
            ):
                collapsed[signature] = mem
        merged = list(collapsed.values())
    recalled = _select_recalled(merged, intent)
    # 台帳の出典は、スコア順の選抜から漏れても必ず載せる（score が付かない経路なので、
    # そのままでは下位に沈む）。年表は時系列に並べ直されるので、追加しても
    # 「先頭＝最古」は崩れない。
    shown_keys = {
        str(mem.get("ts") or "") or f"{mem.get('user_text')}\n{mem.get('reply_text')}"
        for mem in recalled
    }
    for mem in ledger_turns:
        key = str(mem.get("ts") or "") or (
            f"{mem.get('user_text')}\n{mem.get('reply_text')}"
        )
        if key not in shown_keys:
            recalled.append(mem)
            shown_keys.add(key)
    result["stats"]["shown"] = len(recalled)

    blocks: list[str] = []
    block_mode: list[str] = []
    if recalled:
        # 列挙質問も年表（古い順・日付つき）で出す。スコア順にバラバラだと LLM が
        # 重複を畳みにくく、時期の重なりも判断できない。
        if temporal or since or enum:
            span = rag_memory.memory_span(character_id, slot=slot, mode=mode)
            blocks.append(rag_memory.build_timeline_block(recalled, span=span))
            block_mode.append("timeline")
        else:
            blocks.append(rag_memory.build_memory_block(recalled))
    if facts:
        ledger_block = rag_memory.build_ledger_block(facts)
        if ledger_block:
            blocks.append("【記録から拾った出来事の一覧】\n" + ledger_block)
            block_mode.append("ledger")
    result["block"] = "\n\n".join(block for block in blocks if block)
    result["mode"] = "+".join(block_mode)
    return result


def make_fact_llm(model: str | None = None, generation_mode: str = "") -> object:
    """fact_extract へ注入する LLM 呼び出し（system, user → 本文）を作る。

    抽出は短い JSON を返させるだけなので上限も小さく、思考は抑止する。
    失敗時は例外を投げ、呼び出し側（fact_extract.extract）がルール結果へ落ちる。
    """

    def call(system: str, prompt: str) -> str:
        payload = {
            "model": model or DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": _FACT_EXTRACT_MAXTOK,
            "stream": False,
        }
        mode = str(generation_mode or "").strip().lower()
        if mode not in LM_GENERATION_MODES:
            mode = DEFAULT_LM_GENERATION_MODE
        data = _request_lmstudio_content(
            payload,
            segmented_mode=False,
            auto_emoji=False,
            base_max_tokens=_FACT_EXTRACT_MAXTOK,
            mode=mode,
            # 短い JSON 配列を返すだけの抽出器なので、思考ONモードでも枠は貸さない。
            cap_to_base_max_tokens=True,
        )
        return str(data["choices"][0]["message"].get("content") or "")

    return call


def extract_facts_for_turn(
    character_id: str,
    source_id: int,
    *,
    user_text: str,
    reply_text: str,
    ts: str = "",
    slot: str = "main",
    mode: str = "normal",
    speaker: str = "",
    user_name: str = "",
    char_name: str = "",
    model: str | None = None,
    generation_mode: str = "",
    use_llm: bool = True,
) -> int:
    """1 往復を事実台帳へ抽出・保存する（増分抽出。別スレッドから呼ぶ）。

    ルールで主客が確定した往復は LLM を呼ばずに終わる（大半はここで済む）。
    保存した事実の数を返す。例外は投げず 0 を返す（会話本体に影響させない）。
    """
    if rag_memory is None or fact_extract is None:
        return 0
    try:
        llm = make_fact_llm(model, generation_mode) if use_llm else None
        facts = fact_extract.extract(
            user_text,
            reply_text,
            user_name=user_name,
            char_name=char_name,
            mode=mode,
            speaker=speaker,
            ts=ts,
            llm=llm,
        )
        if not facts:
            return 0
        saved = rag_memory.save_facts(
            character_id,
            facts,
            source_id=source_id,
            ts=ts,
            slot=slot,
            mode=mode,
        )
        if saved:
            print(
                f"[ledger] +{saved} facts from turn {source_id}: "
                + " / ".join(
                    f"{f.get('subject') or '?'}→{f.get('recipient') or '?'} "
                    f"{f.get('verb')}:{f.get('object')}"
                    f"[{f.get('modality') or 'done'}"
                    + (f"@{f.get('occurred')}" if f.get("occurred") else "")
                    + "]"
                    for f in facts[:3]
                )
            )
        return saved
    except Exception as exc:
        print(f"[ledger] extract failed: {type(exc).__name__}: {exc}")
        return 0


# --- 思考(reasoning)タグの除去 -------------------------------------------------
# 思考をどう囲むかはモデル系列ごとに違う。書式を知っているサーバなら思考は
# reasoning_content 側へ分離されて content には出ないが、未知の書式だと content へ
# そのまま混ざる。混ざったまま TTS へ渡すと「ユーザーは挨拶している」のような独白を
# 読み上げてしまうので、本文として使う前にここで落とす。
# 書式そのものは LM_MODEL_CATALOG の "thinking" が指す LM_THINKING_FORMATS に置いてある
# （モデルを足す人が「名前だけ足せば済む」と誤解しないよう、モデルと書式を同じ場所に並べる）。


def _build_thinking_patterns() -> tuple[str, str]:
    """カタログに載っている全系列の思考タグを OR で束ねた (開始, 終了) を返す。

    どのモデルが答えたかはリクエスト単位でしか分からない（画面のプルダウンでカタログ外の
    モデルも選べる）ので、書式はモデルで切り替えず、既知の書式すべての和で判定する。
    書式名の綴り違いはここで KeyError にして起動を止める。黙って素通りさせると
    「思考が本文に混ざる」という追いにくい形で表に出るため、編集した直後に気付ける方を採る。
    """
    names: list[str] = []
    for entry in LM_MODEL_CATALOG:
        for name in entry.get("thinking") or ():
            if name not in names:
                names.append(name)
    opens = "|".join(LM_THINKING_FORMATS[name][0] for name in names)
    closes = "|".join(LM_THINKING_FORMATS[name][1] for name in names)
    return f"(?:{opens})", f"(?:{closes})"


_THINK_OPEN, _THINK_CLOSE = _build_thinking_patterns()
_THINK_BLOCK_RE = re.compile(_THINK_OPEN + r".*?" + _THINK_CLOSE, re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(_THINK_OPEN, re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(_THINK_CLOSE, re.IGNORECASE)
if fact_extract is not None:
    # 事実抽出も同じ書式で思考を落とす。カタログを直せば両方に効くよう、ここで配る。
    fact_extract.set_thinking_pattern(_THINK_BLOCK_RE)


def strip_thinking_markup(text: str) -> str:
    """思考タグとその中身を落として本文だけを返す。

    タグが 1 つも無ければ入力をそのまま返す（従来の返答が 1 文字も変わらないことを保証する）。
    ・開始〜終了が揃っている  → その区間を丸ごと捨てる
    ・開始だけで終端が無い    → 思考が途中で切れた形。その位置以降は全部思考なので捨てる
      （結果が空になるので、呼び出し側の「空返答の救済」がプリフィル撮り直しへ進む）
    ・終端だけが残っている    → 開始タグをサーバが剥がした形。最後の終端より後ろを本文とする
    """
    raw = str(text or "")
    if "<" not in raw:
        return raw
    cleaned = _THINK_BLOCK_RE.sub("", raw)
    open_match = _THINK_OPEN_RE.search(cleaned)
    if open_match:
        cleaned = cleaned[: open_match.start()]
    closes = list(_THINK_CLOSE_RE.finditer(cleaned))
    if closes:
        cleaned = cleaned[closes[-1].end() :]
    if cleaned == raw:
        return raw
    return cleaned.strip()


def _normalize_choice_content(data: dict) -> dict:
    """レスポンスの本文から思考タグを落として書き戻す（無ければ何もしない）。

    LM への全リクエストは _request_lmstudio_content を通るので、ここ 1 箇所で
    返答本文・検索クエリ書き換え・記憶ダイジェスト・事実抽出のすべてを面で守れる。
    """
    try:
        message = data["choices"][0]["message"]
    except Exception:
        return data
    content = str(message.get("content") or "")
    stripped = strip_thinking_markup(content)
    if stripped != content:
        message["content"] = stripped
        print(f"[lm] stripped thinking markup: {len(content)} -> {len(stripped)} chars")
    return data


def _thinking_prefill(segmented_mode: bool, auto_emoji: bool) -> tuple[str, str]:
    """思考(reasoning)を抑止するためのアシスタント・プリフィルを返す。

    reasoning モデルは assistant 発話の冒頭で思考を吐き始めるため、その冒頭を
    こちらが先に埋めておくと思考フェーズをスキップして本文から書き始める。
    戻り値は (prefill_content, reattach_prefix)。
    ・JSON モードでは書き出しを与え、後で本文へ前置きし直して有効な JSON に戻す。
    ・通常モードは空白1文字だけを与える（応答には echo されないので本文は汚れない）。
    プリフィルは非思考モデルには無害（続きを書くだけ）で、モデル横断で使える。
    """
    if segmented_mode:
        prefix = '{"segments":[{"text":"'
        return prefix, prefix
    if auto_emoji:
        prefix = '{"text":"'
        return prefix, prefix
    return " ", ""


def _request_lmstudio_content(
    payload: dict,
    segmented_mode: bool,
    auto_emoji: bool,
    base_max_tokens: int,
    mode: str,
    cap_to_base_max_tokens: bool = False,
    aux: bool | None = None,
) -> dict:
    """生成モードに応じて LM Studio へチャットし、本文入りの data を返す。

    mode（LM_GENERATION_MODES のいずれか）:
      prefill       : プリフィルで思考を抑止（高速・低品質）。空/テンプレート拒否時は
                      従来仕様（プリフィル無し・base_max_tokens）へフォールバック。
      original      : 従来仕様。base_max_tokens をそのまま使う（思考モデルでは空になり得る）。
      quality_guard : プリフィル無し＋大きな上限 LM_QUALITY_GUARD_MAX_TOKENS（暴走はここで打ち切り）。
      unlimited     : プリフィル無し＋max_tokens=-1（上限なし。思考を必ず完走させる）。
    どのモードでも choices[0].message.content に本文が入った data を返す。

    cap_to_base_max_tokens: 返答本文ではない補助生成（検索クエリ・事実抽出など）で使う。
      quality_guard / unlimited は「返答の思考を絶対に途中で切らない」ための枠なので、
      短い JSON や 1 行クエリを出すだけの呼び出しに同じ枠を与えると、モデルが枠いっぱい
      まで書き続ける。実測では max_tokens=256 指定の事実抽出が quality_guard の 8192 に
      上書きされ、2300 トークン超を生成して後処理だけで数分 GPU を占有していた。True の
      間は呼び出し側が申告した base_max_tokens を上限として守る（枠不足で本文が空に
      なった場合は、下の救済がプリフィルで撮り直すので黙って失敗はしない）。

    aux: この呼び出しを補助生成として扱うか（プロンプト KV キャッシュを残させない）。
      None なら cap_to_base_max_tokens に追従する。プリフィル固定の要点メモのように
      枠の頭打ちが要らない補助生成では True を明示する。
    """
    is_aux = cap_to_base_max_tokens if aux is None else bool(aux)
    if mode == "prefill":
        prefill_content, reattach = _thinking_prefill(segmented_mode, auto_emoji)
        # プリフィル（構造化出力は役割が重複し競合しうるので付けない）。
        try:
            prefilled = dict(payload)
            prefilled["messages"] = list(payload["messages"]) + [
                {"role": "assistant", "content": prefill_content}
            ]
            # 思考タグは reattach の前に落とす（本文の前に思考が挟まったまま
            # `{"segments":[{"text":"` を前置きすると JSON が壊れる）。
            data = _normalize_choice_content(
                _post_lmstudio_chat(prefilled, use_structured=False, aux=is_aux)
            )
            msg = data["choices"][0]["message"]
            content = str(msg.get("content") or "")
            if content.strip():
                if reattach:
                    stripped = content.lstrip()
                    # モデルがプリフィルを継続せず、自前で完全な JSON を出した場合は
                    # 前置きすると壊れるので、そのまま採用する。
                    msg["content"] = stripped if stripped.startswith("{") else reattach + content
                else:
                    msg["content"] = content.lstrip()
                return data
        except urllib.error.HTTPError:
            # テンプレートが末尾 assistant を受け付けないモデル → 従来仕様へフォールバック。
            pass
        body = dict(payload)
        body["max_tokens"] = base_max_tokens
        return _normalize_choice_content(
            _post_lmstudio_chat(body, _use_structured_output(segmented_mode), aux=is_aux)
        )

    body = dict(payload)
    if cap_to_base_max_tokens:
        # 補助生成は申告どおりの上限で頭打ちにする（quality_guard/unlimited の枠を貸さない）。
        body["max_tokens"] = base_max_tokens
    elif mode == "unlimited":
        body["max_tokens"] = -1
    elif mode == "quality_guard":
        body["max_tokens"] = LM_QUALITY_GUARD_MAX_TOKENS
    else:  # "original"
        body["max_tokens"] = base_max_tokens
    data = _normalize_choice_content(
        _post_lmstudio_chat(body, _use_structured_output(segmented_mode), aux=is_aux)
    )
    if _choice_content(data).strip():
        return data
    # ここから空返答の救済。典型は「文脈の残り枠を思考が食い切った」ケースで、
    # max_tokens をいくら緩めても直らない（残り枠は文脈長で決まる）。
    _log_empty_reply(data, mode)
    # 1) 思考を止めて撮り直す。残り枠が数百トークンでも本文なら出せる
    #    （実測: 同じプロンプトで reasoning 209tok→0tok、本文 0字→53字）。
    with contextlib.suppress(Exception):
        retry = _request_lmstudio_content(
            payload, segmented_mode, auto_emoji, base_max_tokens, "prefill", aux=is_aux
        )
        if _choice_content(retry).strip():
            print("[lm] empty content -> prefill retry succeeded")
            return retry
    # 2) 最後の保険。打ち切られた思考の中に本文が書かれていることがあるので拾う。
    salvaged = _salvage_from_reasoning(data, segmented_mode or auto_emoji)
    if salvaged:
        print("[lm] empty content -> salvaged text from reasoning_content")
        data["choices"][0]["message"]["content"] = salvaged
    return data


def _log_empty_reply(data: dict, mode: str) -> None:
    """空返答の切り分け材料（打ち切り理由・実トークン数）をサーバログへ残す。"""
    try:
        choice = data["choices"][0]
    except Exception:
        choice = {}
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    print(
        "[lm] empty assistant content: "
        f"mode={mode} finish={choice.get('finish_reason')} "
        f"prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} "
        f"reasoning={details.get('reasoning_tokens')} "
        f"reasoning_chars={len(str((choice.get('message') or {}).get('reasoning_content') or ''))}"
    )


def _salvage_from_reasoning(data: dict, expect_json: bool) -> str:
    """打ち切られた思考（reasoning_content）から、読み上げ可能な本文だけを拾う。

    思考には「ユーザーは挨拶している」等のメタな独白が混ざるので、そのまま TTS へ
    渡すと事故になる。拾うのは (1) JSON の ``"text"`` フィールド、(2) 思考の終了タグ
    （``</think>`` / ``<channel|>``）より後ろの本文だけに限り、確信が持てなければ何も返さない。
    """
    try:
        reasoning = str(data["choices"][0]["message"].get("reasoning_content") or "")
    except Exception:
        return ""
    if not reasoning.strip():
        return ""
    if expect_json:
        parts: list[str] = []
        for match in re.finditer(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', reasoning):
            with contextlib.suppress(Exception):
                value = str(json.loads(f'"{match.group(1)}"')).strip()
                if value:
                    parts.append(value)
        if parts:
            return " ".join(parts)
    closes = list(_THINK_CLOSE_RE.finditer(reasoning))
    if closes:
        return reasoning[closes[-1].end() :].strip()
    return ""


def _trim_block_middle(block: str, budget_tokens: int) -> str:
    """記憶ブロックを行単位で「中央から」間引き、予算内へ収める。

    先頭と末尾を残すのは、年表が古い順に並んでおり『一番最初』は先頭・『最近』は
    末尾にあるためで、どちら側を落としても request_lmstudio が与える読み方の約束
    （先頭が最古／末尾が最新）が崩れる。省略したことは印を残して伝える。
    """
    lines = str(block or "").splitlines()
    if budget_tokens <= 0 or not lines:
        return ""
    used = estimate_tokens(_MEMORY_OMITTED_MARK)
    head: list[str] = []
    tail: list[str] = []
    low, high = 0, len(lines) - 1
    take_head = True
    while low <= high:
        index = low if take_head else high
        cost = estimate_tokens(lines[index]) + 1
        if used + cost > budget_tokens:
            break
        if take_head:
            head.append(lines[index])
            low += 1
        else:
            tail.append(lines[index])
            high -= 1
        used += cost
        take_head = not take_head
    if low > high:
        return "\n".join(head + list(reversed(tail)))
    return "\n".join(head + [_MEMORY_OMITTED_MARK] + list(reversed(tail)))


def _split_block_for_digest(block: str, room_tokens: int) -> list[str]:
    """記憶ブロックを、1 回の要約リクエストに収まる大きさへ行単位で切り分ける。"""
    chunks: list[str] = []
    current: list[str] = []
    used = 0
    for line in str(block or "").splitlines():
        cost = estimate_tokens(line) + 1
        if current and used + cost > room_tokens:
            chunks.append("\n".join(current))
            current, used = [], 0
        current.append(line)
        used += cost
    if current:
        chunks.append("\n".join(current))
    return chunks


def digest_memory_block(
    memory_block: str,
    *,
    question: str,
    budget_tokens: int,
    context_length: int,
    model: str | None,
    enum_mode: bool,
) -> str:
    """想起テキストを別呼び出しで「要点メモ」へ圧縮する（本体プロンプトの枠を空ける）。

    生の想起テキスト（年表＋台帳＋読み方指示で数千字）を本体会話へ丸ごと渡すと、
    思考ぶんの枠が残らず空返答になる。そこで**枠に収まらないときだけ**記憶を読む
    LLM 呼び出しを 1 回挟み、本体には要点だけを渡す。記憶自体が 1 リクエストに
    収まらない場合は分割して要約し、連結する（map-reduce）。

    列挙質問では「圧縮」が網羅性を壊す（要約は主なものだけに絞る力が働く）ので、
    件数は減らさず 1 件 1 行へ短縮させる。失敗・空なら "" を返し、呼び出し側は
    機械的な間引きへフォールバックする。
    """
    block = str(memory_block or "").strip()
    if not block or budget_tokens <= 0 or context_length <= 0:
        return ""
    enum_rule = (
        "・ユーザーは列挙・網羅を求めている。**項目を絶対に減らさず**、"
        "1 件 1 行で全部書き出すこと（各行は短くする）。\n"
        if enum_mode
        else "・質問に関係しない記録は落として構わない。\n"
    )
    system = (
        "あなたは会話ログの要約器です。以下の過去ログ抜粋から、質問に答えるために"
        "必要な事実だけを箇条書きで書き出してください。\n"
        "・1 行 1 件、行頭は「- 」。各行 40 字以内。\n"
        "・**誰が誰にしたのか（主体・向き）を必ず保つ**こと。入れ替えは重大な誤りです。\n"
        "・日付・時期が書かれている記録は、その表記のまま行末に残す。\n"
        "・まだしていない予定・約束・願望は行末に「（予定）」と付ける。\n"
        "・抜粋に無いことは書かない（推測・補完をしない）。\n"
        f"{enum_rule}"
        "・前置き・見出し・説明・思考は出さず、箇条書きだけを返す。/no_think"
    )
    max_tokens = max(160, min(LM_MEMORY_DIGEST_MAXTOK, budget_tokens))
    overhead = estimate_tokens(system) + estimate_tokens(question) + 64
    # チャンクは概算のズレを一番受けやすい（弾かれると 1 リクエスト丸ごと無駄）ので、
    # 残り枠の 85% までしか詰めない。
    room = int((context_length - LM_OUTPUT_RESERVE_TOKENS - overhead - max_tokens) * 0.85)
    chunks = _split_block_for_digest(block, room) if room > 400 else [block]
    if len(chunks) > LM_MEMORY_DIGEST_CHUNKS:
        print(
            f"[digest] {len(chunks)} chunks -> first {LM_MEMORY_DIGEST_CHUNKS} only "
            f"({len(chunks) - LM_MEMORY_DIGEST_CHUNKS} chunks dropped)"
        )
        chunks = chunks[:LM_MEMORY_DIGEST_CHUNKS]

    def ask(source: str, depth: int = 0) -> str:
        payload = {
            "model": model or DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"質問:\n{question}\n\n過去ログ抜粋:\n{source}\n\n箇条書き:",
                },
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        # 要約側は思考させない（prefill）。ここで思考させると本体と同じ枠不足を招くうえ、
        # 要点メモに思考の独白が混ざる。プリフィルは HTTP 400 時の従来仕様フォールバック付き。
        try:
            # 補助生成なのでプロンプト KV は残させない（記憶ブロックは毎回中身が変わり
            # 再利用が効かないのに、載ると返答本文が使いたい履歴の接頭辞を追い出す）。
            data = _request_lmstudio_content(
                payload, False, False, max_tokens, "prefill", aux=True
            )
        except LMContextOverflowError:
            # 概算より実トークンが多くて弾かれた。黙って記録を捨てないよう、半分に
            # 割って撮り直す（記憶ブロックは日付・番号の密度で実トークンが揺れる）。
            lines = source.splitlines()
            if depth >= 2 or len(lines) < 4:
                print(f"[digest] chunk still too large at depth {depth} -> dropped")
                return ""
            middle = len(lines) // 2
            print(f"[digest] chunk too large -> split in half (depth {depth + 1})")
            halves = [
                ask("\n".join(lines[:middle]), depth + 1),
                ask("\n".join(lines[middle:]), depth + 1),
            ]
            return "\n".join(text for text in halves if text)
        except Exception as exc:
            print(f"[digest] failed: {type(exc).__name__}: {exc}")
            return ""
        text = _choice_content(data).strip()
        # 箇条書き以外（前置き・思考の残り）が混ざったら、行頭「-」の行だけ拾う。
        bullets = [line.strip() for line in text.splitlines() if line.strip().startswith(("-", "・"))]
        return "\n".join(bullets) if bullets else text

    parts = [text for text in (ask(chunk) for chunk in chunks) if text]
    digest = "\n".join(parts).strip()
    if not digest:
        return ""
    # 連結してもまだ枠に収まらないなら、要約の要約を 1 回だけ試す（reduce 段）。
    if estimate_tokens(digest) > budget_tokens and len(parts) > 1:
        reduced = ask(digest)
        if reduced:
            digest = reduced
    print(
        f"[digest] memory block {estimate_tokens(block)}tok -> "
        f"{estimate_tokens(digest)}tok (budget={budget_tokens}, chunks={len(parts)})"
    )
    return digest


def request_lmstudio(
    messages: list[dict[str, str]],
    model: str | None,
    auto_emoji: bool,
    reply_length: str,
    character_prompt: str,
    user_address: str,
    no_dialogue: bool = False,
    speaker: str = "リノン",
    two_only_mode: bool = False,
    style_guide: str = "",
    generation_mode: str = DEFAULT_LM_GENERATION_MODE,
    memory_block: str = "",
    memory_mode: str = "",
    memory_enum: bool = False,
) -> tuple[str, str, str, int, list[dict[str, str]]]:
    if generation_mode not in LM_GENERATION_MODES:
        generation_mode = DEFAULT_LM_GENERATION_MODE
    length_instruction, max_tokens, chunk_limit = reply_style_for_length(reply_length)
    address = str(user_address or "").strip() or "あなた"
    address_instruction = (
        f"\nユーザーへの呼びかけは「{address}」を使ってください。"
        "名前が明示されていない相手を「〇〇君」「君」「きみ」と呼ばないでください。"
    )
    no_dialogue_instruction = (
        "\n台詞禁止モードです。呼びかけ、質問、説明、選択肢提示、二人称の使用、"
        "普通の文章として読める会話文は禁止です。発声・吐息・擬音・短い断片だけで構成し、"
        "意味のある文を続けないでください。"
        if no_dialogue
        else ""
    )
    two_only_instruction = (
        "\n2人だけモードです。この会話世界にユーザーや観客は存在しません。"
        "ユーザー入力は登場人物の発言ではなく、外部からの進行指示/お題として扱ってください。"
        "リノンとルヴィアだけが同じ場にいて、互いにだけ話します。"
        "ユーザーへ話しかけたり、ユーザーの反応を求めたり、外部の相手を「きみ」「君」「あなた」などで呼ばないでください。"
        "返答は必ず相手キャラクターへの発言として書いてください。"
        if two_only_mode
        else ""
    )
    # RAG で取り出した過去ログの差し込み枠。関係ある記憶だけ自然に織り込ませ、
    # 無関係な古い記憶は無視させる指示をセットで与える（人格・文脈の統一性維持）。
    # 実際にどれを使うかは、下のプロンプト予算（思考＋本文ぶんの確保）で決める。
    def build_memory_instruction(block: str) -> str:
        block = str(block or "").strip()
        if not block:
            return ""
        memory_instruction = (
            "\n\n【参考：過去の二人の会話の記憶】\n"
            f"{block}\n"
            "これは実際にあった二人の過去の会話の記録で、事実として扱ってよい参考情報です。"
            "各行の「ユーザー:」はユーザーの発言、「返答:」はあなた側の発言です"
            "（どちらが何をしたのかを取り違えないでください）。"
            "関係がありそうなら自然に返答へ活かし、関係がなければ触れないでください。"
            "「記憶によると」「質問の意図を分析すると」等のメタ発言はせず、いつも通り自然に"
            "一人の発言として返してください。ふだんは記憶を機械的に列挙・引用しないこと。"
            "ただしユーザーが『全部挙げて』『他には？』『いくつも』等と明示的に列挙・網羅を"
            "求めた場合に限り、上の記憶から該当する項目を“主なものだけ”に絞らず、思い出せる"
            "限り漏れなく（重複は1つにまとめて）挙げてください。"
        )
        # 時系列で並べた年表を渡した場合の読み方。日付の差分計算は LLM が苦手なので、
        # 計算済みの経過期間をそのまま使わせ、記録範囲の外は断定させない。
        if "timeline" in str(memory_mode or ""):
            memory_instruction += (
                "\n上の記憶は日時つきで**古い順**に並んでいます（先頭が最も古く、末尾が最も新しい）。"
                "『一番最初』『初めて』を訊かれたら該当する最も古い記録を、『最後』『最近』なら"
                "最も新しい記録を答えてください。日付はその話をした日です"
                "（過去を振り返って話した記録では、出来事そのものはもっと前に起きています）。"
                "『いつ？』には添えてある年月と経過期間（例: 約1年前）をそのまま使い、"
                "自分で日数を計算し直さないでください。"
                "『記録の範囲』より前のことは記録が残っていないだけで、"
                "「そんな出来事は無かった」ことにはなりません。"
                "該当が無いときや、範囲の先頭に近すぎて本当に最初か確信が持てないときは、"
                "断定せず『たしか…だったと思う』のように曖昧に答えてください"
                "（覚えていない事実を作らないこと）。"
            )
        # 事実台帳（一覧）を渡した場合の読み方。主客の取り違えを防ぐのが主目的。
        if "ledger" in str(memory_mode or ""):
            memory_instruction += (
                "\n【出来事の一覧】は上の会話記録から機械的に抜き出した索引です。"
                "各行は「誰が→誰に 何をした: 対象」の形で、誰がした事なのかを必ずその通りに"
                "扱ってください（あなたがした事とユーザーがした事を絶対に入れ替えないこと）。"
                "『（主客不明）』と付いた行は、どちらがした事か記録から判別できなかったものなので、"
                "自分がした事として語らないでください。"
                "『＜まだしていない事＞』の節にある行は、**まだ実際にはしていない**"
                "予定・約束・願望や、しなかった事です。『作った物』『行った場所』を訊かれたときに"
                "そこから挙げてはいけません（訊かれたら「今度〜する約束」として話してください）。"
                "『（実行したか不明）』と付いた行も、実際にしたと断定しないでください。"
                "各行の日付は、時期が分かっているものは**出来事そのものの時期**、"
                "分からないものはその話をした日です。"
                "一覧は索引にすぎないので、内容が上の会話記録と食い違うときは会話記録の方を信じてください。"
                "列挙を求められたら一覧の項目を漏れなく挙げ、求められていなければ一覧を読み上げず、"
                "会話の流れに合う分だけ自然に触れてください。"
            )
        return memory_instruction

    def build_digest_instruction(digest: str) -> str:
        """要点メモ（digest_memory_block の出力）用の短い読み方指示。

        生の想起テキスト向けの長い指示（年表の並び順・台帳の読み方）はメモには当て
        はまらないうえ 1400 字近くあって枠を食う。メモ用は主客と予定の扱いだけ伝える。
        """
        digest = str(digest or "").strip()
        if not digest:
            return ""
        return (
            "\n\n【参考：過去の二人の会話の記憶（要点メモ）】\n"
            f"{digest}\n"
            "これは実際にあった二人の過去の会話から、いまの話題に関係する部分だけを"
            "書き出したメモで、事実として扱ってよい参考情報です。"
            "各行が**誰のした事なのか**を必ずその通りに扱ってください"
            "（あなたがした事とユーザーがした事を絶対に入れ替えないこと）。"
            "行末に『（予定）』とあるものは、まだ実際にはしていない予定・約束・願望です"
            "（した事として語らないでください）。"
            "「記憶によると」等のメタ発言はせず、いつも通り自然に一人の発言として"
            "返してください。関係がありそうなら自然に活かし、関係がなければ触れないこと。"
            + (
                "ユーザーは列挙・網羅を求めているので、メモの項目を漏れなく"
                "（重複は1つにまとめて）挙げてください。"
                if memory_enum
                else ""
            )
        )

    # 感情の変化ごとにセグメント分割させる新モード。台詞禁止モード時は挙動維持のため
    # 従来の「返答全体で絵文字1つ」を使う。
    segmented_mode = auto_emoji and not no_dialogue
    # style は最終的に TTS caption へ連結され、リファレンス話者の音色と競合しうる。
    # 既定では声色/音程/年齢/声質を指す語を style に書かせない（音色ドリフト対策）。
    # キャラ別の styleGuide があれば、その指針を追加で強制する。
    style_rule = (
        "・styleは声の「感情・話し方」だけを短い日本語で書いてください"
        "（例:「楽しげにはしゃいで」「静かに真剣な調子で」「怒って責めるように」）。\n"
        "・styleに声色・音程・声質・年齢を指す語（低い声・高い声・かすれ声・幼い声・大人っぽい声・"
        "ハスキー・ささやき声など）は書かないでください。音色はリファレンス音声に任せます。\n"
    )
    style_guide_rule = ""
    if str(style_guide or "").strip():
        style_guide_rule = (
            "・このキャラクター固有の感情表現の指針に必ず従ってください:\n"
            f"{str(style_guide).strip()}\n"
        )
    emoji_instruction = ""
    if segmented_mode:
        emoji_instruction = (
            "\n返答は感情や口調が変わるところで区切り、各セグメントに感情情報を付けてください。"
            "必ず次のJSONだけで返してください:\n"
            '{"segments":[{"text":"発話本文","style":"日本語での感情や口調の指示","emoji":"該当する発声効果の絵文字1つ、なければ空文字"}]}\n'
            "・textはそのまま読み上げる本文です。\n"
            f"{style_rule}"
            f"{style_guide_rule}"
            "・感情の変化が無ければsegmentsは1つでも構いません。返答全体を無理に細切れにしないでください。\n"
            "・各セグメントは必ず text / style / emoji の3キーをこの順で含めてください。"
            "該当する発声効果が無くても emoji は必ず \"emoji\":\"\" と空文字で入れてください（キー名を省略しない）。\n"
            "・出力は有効なJSONのみ。末尾カンマや説明文、コードフェンスは付けないでください。\n"
            "・emojiは次のリストのいずれかだけ使用可。該当する発声効果が無ければ空文字にしてください:\n"
            f"{build_emoji_choice_prompt()}"
        )
    elif auto_emoji:
        emoji_instruction = (
            "\nIrodori-TTSの感情/発声スタイル絵文字を1つだけ選んでください。"
            "自然な通常発話ならemojiは空文字にしてください。"
            "選べる絵文字:\n"
            f"{build_emoji_choice_prompt()}\n"
            '必ずJSONだけで返してください: {"text":"返答本文","emoji":"絵文字または空文字"}'
        )

    def system_content(memory_part: str) -> str:
        return (
            "あなたは日本語で自然に返す会話相手です。\n"
            f"いま話すキャラクターは「{speaker}」です。\n"
            f"{character_prompt.strip()}\n"
            f"{address_instruction}\n"
            f"{no_dialogue_instruction}\n"
            f"{two_only_instruction}"
            f"{memory_part}\n"
            f"{length_instruction}"
            "思考過程は出さず、最終回答だけを出してください。/no_think"
            f"{emoji_instruction}"
        )

    # --- プロンプト予算: 思考＋本文ぶんの枠を先に確保する ---
    # ここを守らないと、プロンプトが文脈長をほぼ埋めて思考が残り枠を食い切り、
    # 本文 0 トークン（空返答）になる。予算からはみ出すのは実測上ほぼ記憶ブロック
    # （年表＋台帳で数千字）だけなので、削るのもそこに限る。
    memory_block = str(memory_block or "").strip()
    memory_instruction = build_memory_instruction(memory_block)
    context_length = lm_loaded_context_length(model) if memory_block else 0
    if memory_block and context_length > 0:
        fixed_tokens = estimate_tokens(system_content("")) + estimate_messages_tokens(messages)
        budget = context_length - LM_OUTPUT_RESERVE_TOKENS - fixed_tokens
        need = estimate_tokens(memory_instruction)
        if need > budget:
            print(
                f"[ctx] memory {need}tok > budget {budget}tok "
                f"(ctx={context_length} fixed={fixed_tokens} reserve={LM_OUTPUT_RESERVE_TOKENS})"
            )
            digest = ""
            if LM_MEMORY_DIGEST and budget > 240:
                # 記憶を読む LLM 呼び出しを 1 回だけ挟み、本体には要点メモだけ渡す。
                # 指示文ぶんを差し引いた残りが、メモに使える枠。
                digest = digest_memory_block(
                    memory_block,
                    question=next(
                        (
                            str(item.get("content") or "")
                            for item in reversed(messages)
                            if item.get("role") == "user"
                        ),
                        "",
                    ),
                    budget_tokens=max(120, budget - estimate_tokens(build_digest_instruction("x"))),
                    context_length=context_length,
                    model=model,
                    enum_mode=memory_enum,
                )
            # 要約が使えたらメモ＋短い指示、使えなければ生ブロック＋従来の指示で組む。
            use_digest = bool(digest)
            build = build_digest_instruction if use_digest else build_memory_instruction
            memory_body = digest if use_digest else memory_block
            memory_instruction = build(memory_body)
            if estimate_tokens(memory_instruction) > budget:
                # それでも溢れるときは行単位で間引いて必ず枠へ収める（指示文ぶんを除いた残り）。
                room = budget - (
                    estimate_tokens(memory_instruction) - estimate_tokens(memory_body)
                )
                memory_instruction = build(_trim_block_middle(memory_body, room))
                print(f"[ctx] memory trimmed to {estimate_tokens(memory_instruction)}tok")
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system_content(memory_instruction)},
            *messages,
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        data = _request_lmstudio_content(
            payload, segmented_mode, auto_emoji, max_tokens, generation_mode
        )
    except LMContextOverflowError as exc:
        # 概算が外れてプロンプトが文脈長を超えた場合。記憶なしでも返答は返したいので、
        # 記憶ブロックを落として 1 度だけ撮り直す（記憶付きの沈黙より記憶なしの返答）。
        if not memory_instruction:
            raise
        print(f"[ctx] prompt exceeded context -> retry without memory block ({exc})")
        payload["messages"][0]["content"] = system_content("")
        data = _request_lmstudio_content(
            payload, segmented_mode, auto_emoji, max_tokens, generation_mode
        )
    choice_message = data["choices"][0]["message"]
    content = str(choice_message.get("content") or "").strip()
    if not content:
        # プリフィル再試行でも思考からの救済でも本文が取れなかったケース。空文字を
        # TTS へ渡さず、原因の切り分けに使える数値を添えて見えるエラーにする。
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        raise RuntimeError(
            "LM Studio returned empty assistant content "
            f"(finish={data['choices'][0].get('finish_reason')}, "
            f"prompt={usage.get('prompt_tokens')}tok, "
            f"completion={usage.get('completion_tokens')}tok, "
            f"reasoning={details.get('reasoning_tokens')}tok, "
            f"n_ctx={context_length or lm_loaded_context_length(model) or 'unknown'}). "
            "思考が残り枠を食い切った可能性が高いです。LM Studio のロード設定で"
            "Context Length を増やすか、LM_OUTPUT_RESERVE_TOKENS を上げてください。"
        )
    allowed_emojis = {item["emoji"] for item in load_emoji_items()}
    segments: list[dict[str, str]] = []
    if segmented_mode:
        segments = parse_lmstudio_segments(content, allowed_emojis)
    if segments:
        message = "".join(seg["text"] for seg in segments).strip()
        emoji = next((seg["emoji"] for seg in segments if seg["emoji"]), "")
    else:
        message, emoji = parse_lmstudio_reply(content, allowed_emojis) if auto_emoji else (content, "")
        message = strip_irodori_style_marks(message)
        # 保険: セグメント抽出をすり抜けた壊れJSONを、記号ごと読み上げないよう本文だけ救済。
        if segmented_mode and ('"segments"' in message or re.search(r'"text"\s*:', message)):
            recovered = " ".join(
                strip_irodori_style_marks(_json_str_field(obj, "text"))
                for obj in re.findall(r"\{[^{}]*\}", message)
                if '"text"' in obj
            ).strip()
            if recovered:
                message = recovered
        if no_dialogue:
            message = sanitize_no_dialogue_reply(message)
    if not message:
        raise RuntimeError("LM Studio returned only style marks and no speakable text.")
    model_used = data.get("model") or payload["model"]
    return message, model_used, emoji, chunk_limit, segments


def ensure_irodori_module():
    global Irodori_module
    if Irodori_module is not None:
        return Irodori_module
    if not IRODORI_ROOT.exists():
        raise RuntimeError(f"Irodori root not found: {IRODORI_ROOT}")
    sys.path.insert(0, str(IRODORI_ROOT))
    old_cwd = Path.cwd()
    os.chdir(IRODORI_ROOT)
    try:
        import gradio_app_voicedesign as app_vd

        quiet_irodori_watermark_warnings()
        Irodori_module = app_vd
        return Irodori_module
    finally:
        os.chdir(old_cwd)


def new_tts_seed() -> int:
    """1 リプライ分の TTS 生成で共有する乱数 seed を作る。

    返答を文（チャンク）ごとに独立生成すると、seed 未指定（ランダム）では
    チャンクごとに音色の当たりがばらつき「一部の文だけ声が違う」原因になる。
    リプライ単位で seed を固定し全チャンクへ渡すことで、感情（caption/emoji）は
    変えつつ音色の実現を揃える。
    """
    return random.randint(1, 2_147_483_646)


def _seed_raw_value(seed: object) -> str:
    """seed を Irodori の ``seed_raw``（``infer.py --seed`` 相当）へ渡す文字列に整える。

    空文字は「seed 未指定＝ランダム」を意味する（従来挙動）。数値化できない値も
    ランダム扱いにフォールバックする。
    """
    if seed is None or seed == "":
        return ""
    try:
        return str(int(seed))
    except (TypeError, ValueError):
        return ""


def synthesize_sentence(
    text: str,
    index: int,
    steps: int,
    emoji_style: str = "",
    caption: str = "",
    ref_wav: Path | None = None,
    duration_scale: float = 1.0,
    cfg_scale_text: float = IRODORI_CFG_SCALE_TEXT,
    cfg_scale_caption: float = IRODORI_CFG_SCALE_CAPTION,
    cfg_scale_speaker: float = IRODORI_CFG_SCALE_SPEAKER,
    seed: object = "",
) -> dict:
    module = ensure_irodori_module()
    # 合成が始まる前に掃除の予約を取り消す（合成中に発火しても Irodori_lock を取れず
    # 空振りするだけだが、予約を消費して掃除が 1 回遅れるのを避ける）。
    _cancel_vram_release()
    # TTS へ渡す本文のみ英字→カナ化（保存・表示は原文のまま、絵文字=発声効果は変換しない）。
    styled_text = apply_emoji_style(english_to_kana_for_tts(text), emoji_style)
    voice_caption = str(caption or "").strip() or IRODORI_CAPTION
    reference_wav = ref_wav or IRODORI_REF_WAV
    cfg_text = sanitize_cfg_scale(cfg_scale_text, IRODORI_CFG_SCALE_TEXT)
    cfg_caption = sanitize_cfg_scale(cfg_scale_caption, IRODORI_CFG_SCALE_CAPTION)
    cfg_speaker = sanitize_cfg_scale(cfg_scale_speaker, IRODORI_CFG_SCALE_SPEAKER)
    # Irodori へ実際に渡す CFG Scale をコンソールへ出力（キャラ別設定の反映確認用）。
    print(f"[cfg] text={cfg_text} caption={cfg_caption} speaker={cfg_speaker}")
    old_cwd = Path.cwd()
    os.chdir(IRODORI_ROOT)
    try:
        with Irodori_lock:
            runtime = irodori_runtime_settings()
            start = time.perf_counter()
            result = module._run_generation(
                IRODORI_CHECKPOINT,
                runtime["modelDevice"],
                runtime["modelPrecision"],
                runtime["codecDevice"],
                runtime["codecPrecision"],
                styled_text,
                voice_caption,
                str(reference_wav) if reference_wav.exists() else None,
                int(steps),
                1,
                _seed_raw_value(seed),
                "",
                float(duration_scale),
                "linear",
                -1.0,
                "independent",
                cfg_text,
                cfg_caption,
                cfg_speaker,
                "",
                0.0,
                1.0,
                True,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            )
            elapsed = time.perf_counter() - start
    finally:
        os.chdir(old_cwd)
        # TTS はこのプロセス内の GPU で走り、torch が解放済みブロックを抱え続ける。
        # 合成のたびに掃除を予約し直し、返答の全チャンクを出し終えて静まってから
        # 1 回だけ empty_cache() させる（チャンクごとに返すと次の確保で待たされる）。
        if _vram_sweep_enabled():
            _schedule_vram_release()

    detail = str(result[-2])
    match = re.search(r"saved\[1\]:\s*(.+)", detail)
    if not match:
        raise RuntimeError(f"Could not find generated wav path in Irodori result: {detail}")
    wav_path = (IRODORI_ROOT / match.group(1).strip()).resolve()
    out_dir = STATIC_ROOT / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"reply_{int(time.time() * 1000)}_{index:02d}.wav"
    public_path = out_dir / safe_name
    shutil.copy2(wav_path, public_path)
    return {
        "text": text,
        "ttsText": styled_text,
        "caption": voice_caption,
        "reference": str(reference_wav) if reference_wav.exists() else "",
        "emojiStyle": emoji_style,
        "speechRate": "fast" if float(duration_scale) < 0.99 else "normal",
        "durationScale": float(duration_scale),
        "modelDevice": runtime["modelDevice"],
        "modelPrecision": runtime["modelPrecision"],
        "codecDevice": runtime["codecDevice"],
        "codecPrecision": runtime["codecPrecision"],
        "expression": expression_for_emoji(emoji_style),
        "url": f"/generated/{safe_name}",
        "elapsed": round(elapsed, 3),
        "source": str(wav_path),
    }


def synthesize_sentence_remote_luvia(
    text: str,
    index: int,
    steps: int,
    emoji_style: str = "",
    caption: str = "",
    remote_ref_wav: str = "",
    duration_scale: float = 1.0,
    remote_tts_url: str = "",
    cfg_scale_text: float = IRODORI_CFG_SCALE_TEXT,
    cfg_scale_caption: float = IRODORI_CFG_SCALE_CAPTION,
    cfg_scale_speaker: float = IRODORI_CFG_SCALE_SPEAKER,
    seed: object = "",
) -> dict:
    cfg_scale_text = sanitize_cfg_scale(cfg_scale_text, IRODORI_CFG_SCALE_TEXT)
    cfg_scale_caption = sanitize_cfg_scale(cfg_scale_caption, IRODORI_CFG_SCALE_CAPTION)
    cfg_scale_speaker = sanitize_cfg_scale(cfg_scale_speaker, IRODORI_CFG_SCALE_SPEAKER)
    target_url = normalize_remote_tts_url(remote_tts_url) or LUVIA_REMOTE_TTS_URL
    if not target_url:
        return synthesize_sentence_remote_luvia_cli(
            text, index, steps, emoji_style, caption, remote_ref_wav, duration_scale,
            cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, seed,
        )
    # TTS へ渡す本文のみ英字→カナ化（保存・表示は原文のまま、絵文字=発声効果は変換しない）。
    styled_text = apply_emoji_style(english_to_kana_for_tts(text), emoji_style)
    voice_caption = str(caption or "").strip() or IRODORI_CAPTION
    payload = {
        "text": styled_text,
        "caption": voice_caption,
        "steps": max(1, min(120, int(steps))),
        "emojiStyle": emoji_style,
        "refWav": remote_ref_wav or LUVIA_REMOTE_REF_WAV,
        "durationScale": float(duration_scale),
        "cfgScaleText": cfg_scale_text,
        "cfgScaleCaption": cfg_scale_caption,
        "cfgScaleSpeaker": cfg_scale_speaker,
        "seed": _seed_raw_value(seed),
    }
    request = urllib.request.Request(
        f"{target_url}/synthesize",
        data=json_bytes(payload),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=240) as res:
            data = json.loads(res.read().decode("utf-8"))
        audio_bytes = base64.b64decode(str(data["audioBase64"]))
        elapsed = float(data.get("elapsed") or (time.perf_counter() - start))
    except Exception as exc:
        if LUVIA_REMOTE_TTS_HOST and LUVIA_REMOTE_IRODORI_ROOT:
            fallback = synthesize_sentence_remote_luvia_cli(
                text, index, steps, emoji_style, caption, remote_ref_wav, duration_scale,
                cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, seed,
            )
            fallback["remoteServerError"] = str(exc)
            fallback["engine"] = "4090-cli-fallback"
            return fallback
        raise RuntimeError(f"Remote 2P TTS failed at {target_url}: {exc}") from exc

    out_dir = STATIC_ROOT / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"reply_{int(time.time() * 1000)}_{index:02d}_luvia4090.wav"
    public_path = out_dir / safe_name
    public_path.write_bytes(audio_bytes)
    return {
        "text": text,
        "ttsText": styled_text,
        "caption": voice_caption,
        "reference": str(data.get("reference") or LUVIA_REMOTE_REF_WAV),
        "emojiStyle": emoji_style,
        "speechRate": "fast" if float(duration_scale) < 0.99 else "normal",
        "durationScale": float(duration_scale),
        "expression": expression_for_emoji(emoji_style),
        "url": f"/generated/{safe_name}",
        "elapsed": round(elapsed, 3),
        "source": str(data.get("source") or target_url),
        "engine": "remote-server",
    }


def synthesize_sentence_remote_luvia_cli(
    text: str,
    index: int,
    steps: int,
    emoji_style: str = "",
    caption: str = "",
    remote_ref_wav: str = "",
    duration_scale: float = 1.0,
    cfg_scale_text: float = IRODORI_CFG_SCALE_TEXT,
    cfg_scale_caption: float = IRODORI_CFG_SCALE_CAPTION,
    cfg_scale_speaker: float = IRODORI_CFG_SCALE_SPEAKER,
    seed: object = "",
) -> dict:
    cfg_scale_text = sanitize_cfg_scale(cfg_scale_text, IRODORI_CFG_SCALE_TEXT)
    cfg_scale_caption = sanitize_cfg_scale(cfg_scale_caption, IRODORI_CFG_SCALE_CAPTION)
    cfg_scale_speaker = sanitize_cfg_scale(cfg_scale_speaker, IRODORI_CFG_SCALE_SPEAKER)
    seed_value = _seed_raw_value(seed)
    if not (LUVIA_REMOTE_TTS_HOST and LUVIA_REMOTE_IRODORI_ROOT):
        raise RuntimeError("Remote Luvia TTS CLI fallback is not configured")
    # TTS へ渡す本文のみ英字→カナ化（保存・表示は原文のまま、絵文字=発声効果は変換しない）。
    styled_text = apply_emoji_style(english_to_kana_for_tts(text), emoji_style)
    voice_caption = str(caption or "").strip() or IRODORI_CAPTION
    request_id = f"luvia_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}_{index:02d}"
    remote_root = LUVIA_REMOTE_IRODORI_ROOT
    remote_requests = rf"{remote_root}\remote_requests"
    remote_outputs = rf"{remote_root}\outputs\remote_luvia"
    remote_request_path = rf"{remote_requests}\{request_id}.json"
    remote_output_wav = rf"{remote_outputs}\{request_id}.wav"
    request_dir = LOG_ROOT / "remote_tts_requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    request_path = request_dir / f"{request_id}.json"
    request_payload = {
        "text": styled_text,
        "caption": voice_caption,
        "steps": max(1, min(120, int(steps))),
        "ref_wav": remote_ref_wav or LUVIA_REMOTE_REF_WAV,
        "output_wav": remote_output_wav,
        "duration_scale": float(duration_scale),
        "cfg_scale_text": cfg_scale_text,
        "cfg_scale_caption": cfg_scale_caption,
        "cfg_scale_speaker": cfg_scale_speaker,
        "seed": seed_value,
    }
    request_path.write_text(json.dumps(request_payload, ensure_ascii=False), encoding="utf-8")

    def run_command(args: list[str], timeout: int = 180) -> None:
        completed = subprocess.run(
            args,
            cwd=str(APP_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(detail or f"command failed: {' '.join(args)}")

    remote_request_scp = remote_request_path.replace("\\", "/")
    remote_output_scp = remote_output_wav.replace("\\", "/")
    run_command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            LUVIA_REMOTE_TTS_HOST,
            (
                f'cmd /c if not exist "{remote_requests}" mkdir "{remote_requests}" '
                f'& if not exist "{remote_outputs}" mkdir "{remote_outputs}"'
            ),
        ],
        timeout=30,
    )
    run_command(
        [
            "scp",
            "-q",
            str(request_path),
            f"{LUVIA_REMOTE_TTS_HOST}:{remote_request_scp}",
        ],
        timeout=60,
    )

    remote_script = (
        "$ErrorActionPreference='Stop';"
        f"$root='{remote_root}';"
        f"$req='{remote_request_path}';"
        "$r=Get-Content -Raw -Encoding UTF8 $req | ConvertFrom-Json;"
        f"Set-Location '{remote_root}';"
        "& '.\\.venv\\Scripts\\python.exe' 'infer.py' "
        "--hf-checkpoint 'Aratako/Irodori-TTS-600M-v3-VoiceDesign' "
        "--text $r.text "
        "--caption $r.caption "
        "--ref-wav $r.ref_wav "
        "--num-steps $r.steps "
        "--duration-scale $r.duration_scale "
        "--t-schedule-mode linear "
        "--sway-coeff -1 "
        "--cfg-guidance-mode independent "
        "--cfg-scale-text $r.cfg_scale_text "
        "--cfg-scale-caption $r.cfg_scale_caption "
        "--cfg-scale-speaker $r.cfg_scale_speaker "
        # seed は Python 側で整数へ検証済みのためスクリプトへ直接埋め込む（空ならランダム）。
        f"{f'--seed {seed_value} ' if seed_value else ''}"
        "--model-precision bf16 "
        "--codec-precision bf16 "
        "--output-wav $r.output_wav"
    )
    encoded_remote_script = base64.b64encode(remote_script.encode("utf-16le")).decode("ascii")
    start = time.perf_counter()
    run_command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            LUVIA_REMOTE_TTS_HOST,
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            encoded_remote_script,
        ],
        timeout=240,
    )
    elapsed = time.perf_counter() - start

    out_dir = STATIC_ROOT / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"reply_{int(time.time() * 1000)}_{index:02d}_luvia4090.wav"
    public_path = out_dir / safe_name
    run_command(
        [
            "scp",
            "-q",
            f"{LUVIA_REMOTE_TTS_HOST}:{remote_output_scp}",
            str(public_path),
        ],
        timeout=60,
    )
    return {
        "text": text,
        "ttsText": styled_text,
        "caption": voice_caption,
        "reference": remote_ref_wav or LUVIA_REMOTE_REF_WAV,
        "emojiStyle": emoji_style,
        "speechRate": "fast" if float(duration_scale) < 0.99 else "normal",
        "durationScale": float(duration_scale),
        "expression": expression_for_emoji(emoji_style),
        "url": f"/generated/{safe_name}",
        "elapsed": round(elapsed, 3),
        "source": f"{LUVIA_REMOTE_TTS_HOST}:{remote_output_wav}",
        "engine": "4090",
    }


def next_external_speak_id() -> int:
    global External_speak_next_id
    with External_speak_lock:
        External_speak_next_id += 1
        return External_speak_next_id


def publish_external_speak_event(event: dict) -> dict:
    event["id"] = next_external_speak_id()
    event["createdAt"] = time.time()
    with External_speak_lock:
        External_speak_events.append(event)
    return event


def external_speak_events_after(after_id: int) -> list[dict]:
    with External_speak_lock:
        return [event for event in External_speak_events if int(event.get("id") or 0) > after_id]


def _resolve_local_wav(item: dict) -> Path | None:
    """1 つの分割音声 item から、結合に使うローカル WAV ファイルのパスを解決する。

    優先度: 我々が一意な名前で ``static/generated`` に複製した公開ファイル（url 由来）。
    これが無い場合のみ、元の ``source`` パスにフォールバックする。source は Irodori の
    出力パス（ファイル名が再利用され上書きされる恐れがある）やリモートパスのことがあるため、
    一意名で管理している公開コピーを優先することで結合の正確性を担保する。
    """
    if not isinstance(item, dict):
        return None
    url = str(item.get("url") or "")
    if url:
        name = Path(unquote(urlparse(url).path or url)).name
        if name:
            candidate = STATIC_ROOT / "generated" / name
            if candidate.exists():
                return candidate
    raw = item.get("source")
    if raw:
        candidate = Path(str(raw))
        if candidate.exists():
            return candidate
    return None


def _read_wav_chunks(path: Path) -> tuple[bytes, bytes]:
    """WAV(RIFF) ファイルから fmt チャンク本体と data チャンクのペイロードを取り出す。

    Python 標準の ``wave`` モジュールは PCM(フォーマット 1) しか扱えず、Irodori-TTS が
    出力する IEEE float(フォーマット 3) の WAV では ``wave.Error: unknown format: 3`` に
    なる。ここでは RIFF を直接パースすることで、PCM/float いずれのフォーマットでも
    フレームデータを取り出せるようにする。
    """
    raw = path.read_bytes()
    if len(raw) < 12 or raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file: {path}")
    fmt_body: bytes | None = None
    data_payload = bytearray()
    pos = 12
    total = len(raw)
    while pos + 8 <= total:
        chunk_id = raw[pos : pos + 4]
        chunk_size = int.from_bytes(raw[pos + 4 : pos + 8], "little")
        body_start = pos + 8
        body_end = min(body_start + chunk_size, total)
        body = raw[body_start:body_end]
        if chunk_id == b"fmt " and fmt_body is None:
            fmt_body = body
        elif chunk_id == b"data":
            data_payload += body
        # チャンクは 2 バイト境界にパディングされる（宣言サイズが奇数なら +1）
        pos = body_start + chunk_size + (chunk_size & 1)
    if fmt_body is None:
        raise ValueError(f"no fmt chunk in {path}")
    if not data_payload:
        raise ValueError(f"no data chunk in {path}")
    return fmt_body, bytes(data_payload)


def _write_wav(output_path: Path, fmt_body: bytes, data_payload: bytes) -> None:
    """fmt チャンク本体と結合済み data ペイロードから 1 つの WAV(RIFF) を書き出す。"""
    fmt_chunk = b"fmt " + len(fmt_body).to_bytes(4, "little") + fmt_body
    if len(fmt_body) & 1:
        fmt_chunk += b"\x00"

    chunks = fmt_chunk
    # 非 PCM(float 等) では fact チャンク(サンプル数)の付与が推奨される
    audio_format = int.from_bytes(fmt_body[0:2], "little") if len(fmt_body) >= 2 else 1
    block_align = int.from_bytes(fmt_body[12:14], "little") if len(fmt_body) >= 14 else 0
    if audio_format != 1 and block_align:
        sample_length = len(data_payload) // block_align
        chunks += b"fact" + (4).to_bytes(4, "little") + sample_length.to_bytes(4, "little")

    data_chunk = b"data" + len(data_payload).to_bytes(4, "little") + data_payload
    if len(data_payload) & 1:
        data_chunk += b"\x00"
    chunks += data_chunk

    riff = b"RIFF" + (4 + len(chunks)).to_bytes(4, "little") + b"WAVE" + chunks
    output_path.write_bytes(riff)


def combine_wav_files(source_paths: list, output_path: Path) -> Path | None:
    """時系列順に並んだ複数の WAV ファイルを 1 つに結合する。

    各パスを必ず 1 つずつ個別に安全に開き（リストオブジェクトをそのまま渡さない）、
    RIFF を直接パースして data チャンクを時系列順に連結する。PCM(1)・IEEE float(3)
    いずれのフォーマットにも対応する。結合に成功した場合は出力パスを、対象ファイルが
    1 つも無い場合は None を返す。
    """
    parsed: list[tuple[bytes, bytes]] = []
    for raw in source_paths:
        if not raw:
            continue
        candidate = Path(str(raw))
        if not candidate.exists():
            print(f"[combine_wav_files] skip missing source: {candidate}")
            continue
        try:
            parsed.append(_read_wav_chunks(candidate))
        except Exception:
            print(f"[combine_wav_files] skip unreadable wav: {candidate}")
            traceback.print_exc()
            continue
    if not parsed:
        return None

    base_fmt = parsed[0][0]
    combined = bytearray()
    for fmt_body, data_payload in parsed:
        # フォーマットが異なるものを連結すると音声が壊れるためスキップする
        if fmt_body != base_fmt:
            print("[combine_wav_files] skip chunk with mismatched fmt")
            continue
        combined += data_payload
    if not combined:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_wav(output_path, base_fmt, bytes(combined))
    return output_path


def build_combined_audio(
    audios: list,
    text: str = "",
    emoji_style: str = "",
    caption: str = "",
) -> dict | None:
    """分割音声（各 item の source）を 1 つの WAV へ結合し、結合済み音声の dict を返す。

    出力は ``static/generated/reply_<timestamp>_combined.wav``。結合対象となる
    ローカル WAV が無い（例: リモート TTS で source がリモートパス）場合や、
    結合中に例外が発生した場合は None を返し、呼び出し側は分割音声にフォールバックできる。
    """
    if not audios:
        return None
    try:
        source_paths = [_resolve_local_wav(item) for item in audios]
        combined_name = f"reply_{int(time.time() * 1000)}_combined.wav"
        combined_output = STATIC_ROOT / "generated" / combined_name
        combined_path = combine_wav_files(source_paths, combined_output)
        if combined_path is None:
            return None
        return {
            "text": text or "".join(str(item.get("text") or "") for item in audios),
            "ttsText": text,
            "caption": caption,
            "emojiStyle": emoji_style,
            "expression": expression_for_emoji(emoji_style),
            "url": f"/generated/{combined_name}",
            "name": combined_name,
            "elapsed": round(
                sum(float(item.get("elapsed") or 0) for item in audios), 3
            ),
            "source": str(combined_path),
        }
    except Exception:
        # 結合に失敗しても分割音声の再生は継続できるよう、原因のみ出力して握りつぶす。
        print("[build_combined_audio] failed to combine wav files")
        traceback.print_exc()
        return None


def render_reply_audio(
    reply: str,
    segments: list | None,
    *,
    speaker_slot: str = "main",
    tts_caption: str = "",
    steps: int = DEFAULT_CHARACTER_STEPS,
    speech_rate: str = "normal",
    emoji_style: str = "",
    fallback_emoji: str = "",
    chunk_limit: int = 8,
    cfg_scale_text: float = IRODORI_CFG_SCALE_TEXT,
    cfg_scale_caption: float = IRODORI_CFG_SCALE_CAPTION,
    cfg_scale_speaker: float = IRODORI_CFG_SCALE_SPEAKER,
    reference_path: Path = IRODORI_REF_WAV,
    second_reference_path: Path = LUVIA_REF_WAV,
    tts_backend_mode: str = "local",
    second_tts_url: str = "",
) -> dict:
    """返答テキスト（＋感情セグメント）から TTS 音声を合成し、結合済み音声までまとめて返す。

    ``/api/chat`` の初回生成と ``/api/regenerate`` の再生成で共通利用する。LLM 生成は含まず、
    合成部分（読み・絵文字スタイル・キャラ別 CFG/steps/reference）のみを担う。読み辞書やコード
    修正は合成時に自動反映されるため、再生成で読み間違いや表現の不備を後から直せる。
    """
    duration_scale = tts_duration_scale_for_rate(speech_rate)
    effective_emoji = emoji_style or fallback_emoji
    use_second_speaker = speaker_slot == "second"
    reference_wav = second_reference_path if use_second_speaker else reference_path
    use_remote_tts = (
        use_second_speaker
        and tts_backend_mode == "remote"
        and remote_luvia_enabled(second_tts_url)
    )
    remote_reference_wav = (
        remote_ref_for_luvia(reference_wav)
        if use_remote_tts and LUVIA_REMOTE_TTS_HOST and LUVIA_REMOTE_IRODORI_ROOT and LUVIA_REMOTE_REF_WAV
        else (LUVIA_REMOTE_REF_WAV if use_remote_tts else "")
    )
    synthesize = synthesize_sentence_remote_luvia if use_remote_tts else synthesize_sentence
    synth_kwargs = (
        {"remote_ref_wav": remote_reference_wav, "remote_tts_url": second_tts_url}
        if use_remote_tts
        else {"ref_wav": reference_wav}
    )
    # CFG Scale はキャラクターごとの値を全 TTS 経路へ共通で渡す。
    # seed はリプライ単位で 1 つ生成し全チャンクへ共通で渡す（音色の当たりを揃える）。
    synth_kwargs.update(
        {
            "cfg_scale_text": cfg_scale_text,
            "cfg_scale_caption": cfg_scale_caption,
            "cfg_scale_speaker": cfg_scale_speaker,
            "seed": new_tts_seed(),
        }
    )
    # 感情セグメント単位に (感情style, 絵文字, 本文) を組み立てる。
    # segments が空（=分割なし/機能オフ）なら返答全体を 1 セグメントとして扱う。
    if segments:
        seg_units = [
            (
                str(seg.get("style") or ""),
                str(emoji_style or seg.get("emoji") or ""),
                str(seg.get("text") or ""),
            )
            for seg in segments
        ]
    else:
        seg_units = [("", effective_emoji, reply)]
    seg_chunk_limit = max(1, min(20, chunk_limit))
    steps = max(1, min(120, steps))
    audios: list[dict] = []
    chunks: list[str] = []
    seg_meta: list[dict[str, str]] = []
    for seg_style, seg_emoji, seg_text in seg_units:
        # セグメント（感情の単位）は原則 1 発話でまとめて生成する。短い単独チャンクを
        # 作らないことで TTS の末尾暴走を防ぐ。長すぎるセグメントのみ複数チャンクへ分割。
        seg_chunks = group_sentences(
            seg_text, max_chars=TTS_SEGMENT_MAX_CHARS, limit=seg_chunk_limit
        )
        if not seg_chunks:
            continue
        # 各セグメントの caption は「基底 TTS Caption + 感情style」を必ず連結する。
        seg_caption = compose_caption(tts_caption, seg_style)
        seg_meta.append({"style": seg_style, "emoji": seg_emoji, "text": seg_text})
        for chunk_pos, chunk in enumerate(seg_chunks):
            # 発声効果の絵文字は種類で出し分ける。単発音（吐息・喘ぎ・泣き声など）は
            # 先頭チャンクのみに付与し、繰り返し挿入（＝意味不明な発声）を防ぐ。持続系
            # （話し方・声色・音響効果）は全チャンクに付与し表現のぶれを防ぐ。通常はセグメント
            # ＝1 チャンクなのでどちらも先頭に 1 回。分割された長いセグメントで差が出る。
            chunk_emoji = seg_emoji if (chunk_pos == 0 or emoji_is_sustained(seg_emoji)) else ""
            audios.append(
                synthesize(
                    chunk,
                    len(audios) + 1,
                    steps=steps,
                    emoji_style=chunk_emoji,
                    caption=seg_caption,
                    duration_scale=duration_scale,
                    **synth_kwargs,
                )
            )
            chunks.append(chunk)
    # 代表となる感情絵文字（立ち絵 pose 用）: 最初の非空セグメント絵文字、無ければ従来値。
    representative_emoji = (
        next((meta["emoji"] for meta in seg_meta if meta["emoji"]), "")
        or effective_emoji
    )
    # 分割音声を時系列順に 1 つの WAV へ結合する（ローカル TTS のみ対象）。
    combined_audio = build_combined_audio(
        audios, text=reply, emoji_style=representative_emoji, caption=tts_caption
    )
    return {
        "audios": audios,
        "chunks": chunks,
        "segMeta": seg_meta,
        "combined": combined_audio,
        "representativeEmoji": representative_emoji,
        "expression": expression_for_emoji(representative_emoji),
        "effectiveEmoji": effective_emoji,
        "durationScale": duration_scale,
        "useRemoteTts": use_remote_tts,
        "referenceWav": reference_wav,
    }


def regenerate_reply_audio(payload: dict) -> dict:
    """選択中の返答テキストを、LLM を介さず TTS のみ再合成して差し替える。

    英数字の読み間違いを辞書追加で直したり、合成コードの不備を修正した後に、同じ返答
    テキストへ最新の読み・表現を適用し直すための経路。感情セグメント・キャラ別 CFG/steps/
    reference はフロントが保持している生成時の値をそのまま受け取り、忠実に再現する。
    """
    reply = str(payload.get("reply") or "").strip()
    segments = payload.get("segments") if isinstance(payload.get("segments"), list) else []
    if not reply and not segments:
        raise ValueError("reply text is required")
    speaker_slot = "second" if str(payload.get("speakerSlot") or "") == "second" else "main"
    speaker = str(payload.get("speaker") or ("ルヴィア" if speaker_slot == "second" else "リノン")).strip()
    tts_caption = str(payload.get("ttsCaption") or IRODORI_CAPTION).strip()
    steps = sanitize_steps(payload.get("steps"), DEFAULT_CHARACTER_STEPS)
    speech_rate = str(payload.get("speechRate") or "normal").strip().lower()
    emoji_style = str(payload.get("emojiStyle") or "").strip()
    llm_emoji = str(payload.get("llmEmojiStyle") or "").strip()
    cfg_scale_text = sanitize_cfg_scale(payload.get("cfgScaleText"), IRODORI_CFG_SCALE_TEXT)
    cfg_scale_caption = sanitize_cfg_scale(payload.get("cfgScaleCaption"), IRODORI_CFG_SCALE_CAPTION)
    cfg_scale_speaker = sanitize_cfg_scale(payload.get("cfgScaleSpeaker"), IRODORI_CFG_SCALE_SPEAKER)
    reference_path = sanitize_reference_path(payload.get("referencePath"), IRODORI_REF_WAV)
    second_reference_path = sanitize_reference_path(payload.get("secondReferencePath"), LUVIA_REF_WAV)
    tts_backend_mode = str(payload.get("ttsBackendMode") or "local").strip().lower()
    second_tts_host = str(payload.get("secondTtsHost") or payload.get("secondTtsUrl") or "").strip()
    second_tts_url = normalize_remote_tts_url(second_tts_host)

    # 感情スタイル（絵文字）は生成時に seg_meta へ焼き込まれているため、絵文字上書きは掛けず
    # セグメント側の値をそのまま使う。セグメントが無い返答のみ代表絵文字をフォールバックに使う。
    render = render_reply_audio(
        reply,
        segments,
        speaker_slot=speaker_slot,
        tts_caption=tts_caption,
        steps=steps,
        speech_rate=speech_rate,
        emoji_style="",
        fallback_emoji=emoji_style or llm_emoji,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_caption=cfg_scale_caption,
        cfg_scale_speaker=cfg_scale_speaker,
        reference_path=reference_path,
        second_reference_path=second_reference_path,
        tts_backend_mode=tts_backend_mode,
        second_tts_url=second_tts_url,
    )
    audios = render["audios"]
    combined_audio = render["combined"]
    if not audios:
        raise ValueError("regeneration produced no audio")
    append_chat_log(
        {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": "regenerate",
            "speaker": speaker,
            "speakerSlot": speaker_slot,
            "reply": reply,
            "emojiStyle": render["representativeEmoji"],
            "expression": render["expression"],
            "ttsCaption": tts_caption,
            "reference": str(render["referenceWav"]),
            "secondTtsRemote": render["useRemoteTts"],
            "chunkCount": len(render["chunks"]),
            "combinedUrl": (combined_audio or {}).get("url"),
            "audios": [
                {
                    "text": item.get("text"),
                    "ttsText": item.get("ttsText"),
                    "emojiStyle": item.get("emojiStyle"),
                    "expression": item.get("expression"),
                    "elapsed": item.get("elapsed"),
                    "url": item.get("url"),
                }
                for item in audios
            ],
        }
    )
    return {
        "ok": True,
        "reply": reply,
        "speaker": speaker,
        "segments": render["segMeta"],
        "emojiStyle": render["representativeEmoji"],
        "expression": render["expression"],
        "speechRate": speech_rate,
        "durationScale": render["durationScale"],
        "audios": audios,
        "combined": combined_audio,
    }


def handle_external_speak(payload: dict) -> dict:
    text = str(payload.get("text") or payload.get("message") or "").strip()
    if not text:
        raise ValueError("text is required")
    speaker_slot = "second" if str(payload.get("speakerSlot") or payload.get("slot") or "") == "second" else "main"
    use_second_speaker = speaker_slot == "second"
    speaker = str(payload.get("speaker") or ("ルヴィア" if use_second_speaker else "リノン")).strip()
    emoji_style = str(payload.get("emoji") or payload.get("emojiStyle") or "").strip()
    caption = str(payload.get("caption") or payload.get("ttsCaption") or IRODORI_CAPTION).strip()
    cfg_scale_text = sanitize_cfg_scale(payload.get("cfgScaleText"), IRODORI_CFG_SCALE_TEXT)
    cfg_scale_caption = sanitize_cfg_scale(payload.get("cfgScaleCaption"), IRODORI_CFG_SCALE_CAPTION)
    cfg_scale_speaker = sanitize_cfg_scale(payload.get("cfgScaleSpeaker"), IRODORI_CFG_SCALE_SPEAKER)
    steps = max(1, min(120, int(payload.get("steps") or 12)))
    speech_rate = str(payload.get("speechRate") or "normal").strip().lower()
    duration_scale = float(payload.get("durationScale") or tts_duration_scale_for_rate(speech_rate))
    chunk_limit = max(1, min(20, int(payload.get("chunkLimit") or 8)))
    reference_path = sanitize_reference_path(
        payload.get("referencePath"),
        LUVIA_REF_WAV if use_second_speaker else IRODORI_REF_WAV,
    )
    # 発話は原則 1 発話でまとめて生成（短文暴走の回避）。長すぎる場合のみ複数へ分割。
    chunks = group_sentences(text, max_chars=TTS_SEGMENT_MAX_CHARS, limit=chunk_limit)
    # seed はこの発話全体で共通（音色の当たりをチャンク間で揃える）。
    speak_seed = new_tts_seed()
    # 発声効果の絵文字は種類で出し分け：単発音は先頭 1 文のみ、持続系は全チャンクに付与。
    audios = [
        synthesize_sentence(
            chunk,
            index,
            steps=steps,
            emoji_style=(emoji_style if (index == 1 or emoji_is_sustained(emoji_style)) else ""),
            caption=caption,
            ref_wav=reference_path,
            duration_scale=duration_scale,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_caption=cfg_scale_caption,
            cfg_scale_speaker=cfg_scale_speaker,
            seed=speak_seed,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]

    # 分割音声（各 item の source パス）を時系列順に 1 つの WAV へ結合する。
    combined_audio = build_combined_audio(
        audios, text=text, emoji_style=emoji_style, caption=caption
    )

    event = publish_external_speak_event(
        {
            "source": "external",
            "speaker": speaker,
            "speakerSlot": speaker_slot,
            "text": text,
            "chunks": chunks,
            "audios": audios,
            "combined": combined_audio,
            "emojiStyle": emoji_style,
            "expression": expression_for_emoji(emoji_style),
            "caption": caption,
            "reference": str(reference_path),
            "speechRate": speech_rate,
            "durationScale": duration_scale,
        }
    )
    append_chat_log(
        {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": "externalSpeak",
            "speaker": speaker,
            "speakerSlot": speaker_slot,
            "reply": text,
            "emojiStyle": emoji_style,
            "expression": event["expression"],
            "ttsCaption": caption,
            "reference": str(reference_path),
            "chunkCount": len(chunks),
            "combinedUrl": (combined_audio or {}).get("url"),
            "audios": [
                {
                    "text": item.get("text"),
                    "ttsText": item.get("ttsText"),
                    "emojiStyle": item.get("emojiStyle"),
                    "expression": item.get("expression"),
                    "elapsed": item.get("elapsed"),
                    "url": item.get("url"),
                }
                for item in audios
            ],
        }
    )
    return event


def get_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{LM_STUDIO_URL}/models", timeout=5) as res:
            data = json.loads(res.read().decode("utf-8"))
        return [item["id"] for item in data.get("data", []) if item.get("id")]
    except Exception:
        return []


def model_is_supported(model_id: object) -> bool:
    """画面のプルダウンで選ばせてよいモデルか（カタログに載っている系列か）を返す。"""
    return match_model_catalog(model_id) is not None


def preferred_model_option(models: list[str] | None = None) -> str:
    """環境変数で決まったモデルに対応する「``/models`` が返す ID」を返す。

    ``LM_STUDIO_MODEL`` には GGUF のフルパスが入ることがあり、プルダウンに並ぶ ID
    （LM Studio が返す短い名前）とは書き方が違う。記号を落として突き合わせ、同じものを
    指していると分かればプルダウン側の ID を返す。見つからなければ環境変数の文字列を
    そのまま返す（それ自体が LM Studio に通る有効な指定なので、勝手に別 ID へ寄せない）。
    """
    listed = models if models is not None else get_models()
    target = _model_key(DEFAULT_MODEL)
    best = ""
    for model_id in listed:
        key = _model_key(model_id)
        # 6 文字未満の ID は偶然の部分一致が起きやすいので相手にしない。
        if len(key) < 6 or not (target in key or key in target):
            continue
        # 12b-it と 12b-qat が両方ロードされていると、QAT のファイル名（…-it-QAT-…）が
        # it 側の ID にも当たる。より限定の強い（長い）ID を採る。
        if len(key) > len(_model_key(best)):
            best = model_id
    return best or DEFAULT_MODEL


class Handler(BaseHTTPRequestHandler):
    server_version = "IrodoriLMStudioChat/0.1"

    def send_json(self, status: int, payload: object) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            diagnostics = environment_diagnostics()
            self.send_json(
                200,
                {
                    "lmStudioUrl": LM_STUDIO_URL,
                    "models": diagnostics["models"],
                    "preferredModel": diagnostics["preferredModel"],
                    "supportedModels": diagnostics["supportedModels"],
                    "contextLimit": DEFAULT_CONTEXT_LIMIT,
                    "irodoriRoot": str(IRODORI_ROOT),
                    "irodoriReady": diagnostics["irodoriRootExists"] and diagnostics["irodoriPythonExists"],
                    "checkpoint": IRODORI_CHECKPOINT,
                    "ttsCaption": IRODORI_CAPTION,
                    "reference": str(IRODORI_REF_WAV),
                    "referenceExists": IRODORI_REF_WAV.exists(),
                    "luviaReference": str(LUVIA_REF_WAV),
                    "luviaReferenceExists": LUVIA_REF_WAV.exists(),
                    "userReferenceRoot": str(USER_REFERENCE_ROOT),
                    "diagnostics": diagnostics,
                    "luviaRemoteTtsHost": LUVIA_REMOTE_TTS_HOST,
                    "luviaRemoteTtsUrl": LUVIA_REMOTE_TTS_URL,
                    "luviaRemoteDefaultPort": LUVIA_REMOTE_DEFAULT_PORT,
                    "luviaRemoteReference": LUVIA_REMOTE_REF_WAV,
                    "codexInboxSize": len(Codex_inbox),
                    "emojis": safe_load_emoji_items(),
                    "expressions": expression_assets(),
                },
            )
            return

        if parsed.path == "/api/log-summary":
            self.send_json(200, chat_log_summary())
            return

        if parsed.path == "/api/codex-inbox":
            query = parse_qs(parsed.query)
            after_raw = (query.get("after") or ["0"])[0]
            try:
                after_id = int(after_raw or 0)
            except ValueError:
                after_id = 0
            events = codex_inbox_since(after_id)
            self.send_json(
                200,
                {
                    "events": events,
                    "latestId": events[-1]["id"] if events else after_id,
                },
            )
            return

        if parsed.path == "/api/speak-events":
            query = parse_qs(parsed.query)
            after_raw = (query.get("after") or ["0"])[0]
            if str(after_raw).lower() == "latest":
                self.send_json(
                    200,
                    {
                        "events": [],
                        "latestId": External_speak_next_id,
                    },
                )
                return
            try:
                after_id = int(after_raw or 0)
            except ValueError:
                after_id = 0
            events = external_speak_events_after(after_id)
            self.send_json(
                200,
                {
                    "events": events,
                    "latestId": events[-1]["id"] if events else after_id,
                },
            )
            return

        if parsed.path == "/api/session":
            try:
                profile = load_session_profile()
                # 使うモデルは常に環境変数（＝起動スクリプト）が決める。保存された選択を
                # そのまま返すと、画面が /api/status で選んだモデルを後から上書きしてしまい、
                # スクリプトを差し替えても前回のモデルが呼ばれ続ける。保存されたファイル自体
                # には触れないので、将来ここを保存値優先へ戻すこともできる。
                profile["settings"] = {
                    **profile["settings"],
                    "model": preferred_model_option(),
                }
                self.send_json(200, profile)
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return

        if parsed.path == "/api/characters":
            try:
                self.send_json(200, load_character_profiles())
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return

        path = "/index.html" if parsed.path == "/" else parsed.path
        if parsed.path in {"/README.md", "/README.ja.md"}:
            file_path = (APP_ROOT / parsed.path.lstrip("/")).resolve()
            if not str(file_path).startswith(str(APP_ROOT.resolve())) or not file_path.exists():
                self.send_error(404)
                return
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        rel = Path(unquote(path.lstrip("/")))
        static_root = STATIC_ROOT.resolve()
        file_path = (STATIC_ROOT / rel).resolve()
        if parsed.path.startswith("/saved_audio/"):
            static_root = SAVED_AUDIO_ROOT.resolve()
            rel_audio = Path(unquote(parsed.path.removeprefix("/saved_audio/")))
            file_path = (SAVED_AUDIO_ROOT / rel_audio).resolve()
        if parsed.path.startswith("/Character/"):
            static_root = CHARACTER_ROOT.resolve()
            rel_character = Path(unquote(parsed.path.removeprefix("/Character/")))
            file_path = (CHARACTER_ROOT / rel_character).resolve()
        if parsed.path.startswith("/characters/"):
            static_root = LEGACY_CHARACTER_ROOT.resolve()
            rel_character = Path(unquote(parsed.path.removeprefix("/characters/")))
            file_path = (LEGACY_CHARACTER_ROOT / rel_character).resolve()
        if not str(file_path).startswith(str(static_root)) or not file_path.exists():
            self.send_error(404)
            return
        mime, _ = mimetypes.guess_type(str(file_path))
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/session":
            try:
                profile = save_session_profile(read_json_body(self))
                self.send_json(
                    200,
                    {
                        "ok": True,
                        "path": str(SESSION_PROFILE_PATH),
                        "savedAt": profile["savedAt"],
                        "historyCount": sum(len(v) for v in profile["histories"].values()),
                    },
                )
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return

        if parsed.path == "/api/characters":
            try:
                self.send_json(200, save_character_profiles(read_json_body(self)))
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return

        if parsed.path == "/api/delete-turn":
            # UI で削除した往復を、大本の生ログ・感情ログ・RAG検索DB からも取り除く。
            # 履歴の自動保存（/api/session）とは別経路にしてある: 自動保存は「いまの会話
            # コンテキスト」を書くだけなので、そこから削除を推論すると
            # Clear Context（コンテキストのリセット）と区別できず、生ログを消してしまう。
            try:
                self.send_json(200, delete_turn_records(read_json_body(self)))
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return

        if parsed.path == "/api/save-audio":
            try:
                self.send_json(200, save_current_audio(read_json_body(self)))
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return

        if parsed.path == "/api/reference":
            try:
                self.send_json(200, save_reference_audio(read_json_body(self)))
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return

        if parsed.path == "/api/character-image":
            try:
                self.send_json(200, save_character_image(read_json_body(self)))
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return

        if parsed.path == "/api/regenerate":
            try:
                self.send_json(200, regenerate_reply_audio(read_json_body(self)))
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return

        if parsed.path == "/api/speak":
            try:
                event = handle_external_speak(read_json_body(self))
                self.send_json(200, {"ok": True, "event": event})
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return

        if parsed.path == "/api/shutdown":
            try:
                stopped = stop_irodori_ui_processes()
                self.send_json(200, {"ok": True, "stopped": stopped, "self": os.getpid()})
                shutdown_app_server(self.server)
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return

        if parsed.path == "/api/codex-inbox":
            try:
                if self.command == "POST":
                    body = read_json_body(self)
                    message = str(body.get("message") or "").strip()
                    if not message:
                        self.send_json(400, {"error": "message is required"})
                        return
                    item = enqueue_codex_inbox(
                        {
                            "source": str(body.get("source") or "ui").strip(),
                            "message": message,
                            "speaker": str(body.get("speaker") or "").strip(),
                            "speakerSlot": str(body.get("speakerSlot") or "").strip(),
                            "model": str(body.get("model") or "").strip(),
                            "systemPrompt": str(body.get("systemPrompt") or "").strip(),
                            "ttsCaption": str(body.get("ttsCaption") or "").strip(),
                            "history": body.get("history") if isinstance(body.get("history"), list) else [],
                            "contextStats": body.get("contextStats") if isinstance(body.get("contextStats"), dict) else {},
                            "webContext": str(body.get("webContext") or "").strip(),
                            "webTopic": str(body.get("webTopic") or "").strip(),
                            "replyLength": str(body.get("replyLength") or "").strip(),
                            "speechRate": str(body.get("speechRate") or "").strip(),
                            "emojiStyle": str(body.get("emojiStyle") or "").strip(),
                        }
                    )
                    self.send_json(200, {"ok": True, "event": item})
                    return
                after = int(parse_qs(parsed.query).get("after", ["0"])[0] or 0)
                events = codex_inbox_since(after)
                latest = events[-1]["id"] if events else Codex_inbox_next_id
                self.send_json(200, {"events": events, "latestId": latest})
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return

        if parsed.path != "/api/chat":
            self.send_error(404)
            return
        try:
            body = read_json_body(self)
            user_text = str(body.get("message", "")).strip()
            if not user_text:
                self.send_json(400, {"error": "message is required"})
                return
            history = body.get("history") if isinstance(body.get("history"), list) else []
            model = str(body.get("model") or "").strip() or None
            # steps は body（フロントがキャラ別値を解決して送る）を優先。未指定はキャラ別既定へ。
            steps = sanitize_steps(body.get("steps"), DEFAULT_CHARACTER_STEPS)
            speech_rate = str(body.get("speechRate") or "normal").strip().lower()
            duration_scale = tts_duration_scale_for_rate(speech_rate)
            emoji_style = str(body.get("emojiStyle") or "").strip()
            auto_emoji = bool(body.get("autoEmoji", True))
            no_dialogue = bool(body.get("noDialogue", False))
            reply_length = str(body.get("replyLength") or "normal").strip()
            generation_mode = str(body.get("llmGenerationMode") or "").strip().lower()
            if generation_mode not in LM_GENERATION_MODES:
                generation_mode = DEFAULT_LM_GENERATION_MODE
            speaker_slot = "second" if str(body.get("speakerSlot") or "") == "second" else "main"
            speaker = str(body.get("speaker") or ("ルヴィア" if body.get("twoPlayerMode") else "リノン")).strip()
            # RAG 長期記憶はキャラクター単位で分離する。フロントは characterId を送るが、
            # 未指定の旧クライアント互換として speaker 名からも導出する。
            character_id = safe_character_id(body.get("characterId") or speaker)
            if model == "__codex_queue__":
                queued = enqueue_codex_inbox(
                    {
                        "source": "chat",
                        "message": user_text,
                        "speaker": speaker,
                        "speakerSlot": speaker_slot,
                        "model": model,
                        "systemPrompt": str(body.get("systemPrompt") or "").strip(),
                        "ttsCaption": str(body.get("ttsCaption") or "").strip(),
                        "history": history,
                        "contextStats": {},
                        "webContext": str(body.get("webContext") or "").strip(),
                        "webTopic": str(body.get("webTopic") or "").strip(),
                        "replyLength": reply_length,
                        "speechRate": speech_rate,
                        "emojiStyle": emoji_style,
                        "llmGenerationMode": generation_mode,
                    }
                )
                self.send_json(
                    200,
                    {
                        "reply": f"Codex queue に送ったよ。id={queued['id']}",
                        "speaker": speaker,
                        "model": "Codex queue",
                        "chunks": [f"Codex queue に送ったよ。id={queued['id']}"],
                        "emojiStyle": "",
                        "expression": "broadcast",
                        "llmEmojiStyle": "",
                        "autoEmoji": False,
                        "replyLength": reply_length,
                        "speechRate": speech_rate,
                        "durationScale": duration_scale,
                        "audios": [],
                        "contextStats": {},
                        "webSearch": False,
                        "twoOnlyMode": False,
                        "webQuery": "",
                        "webContext": "",
                        "webResults": [],
                        "codexQueued": True,
                        "codexQueueId": queued["id"],
                    },
                )
                return
            # 会話中に画面のプルダウンで替えたモデルはここから有効になる。ただし対象外の
            # 名前（古い保存値・旧クライアント）はそのまま呼ばず、環境変数側の既定へ落とす。
            # Codex queue は LM への呼び出しではないので、上の分岐を抜けた後で判定する。
            model = resolve_lm_model(model, log=False)
            two_player_mode = bool(body.get("twoPlayerMode", False))
            two_only_mode = bool(body.get("twoOnlyMode", False)) and two_player_mode
            use_second_speaker = speaker_slot == "second"
            use_web_search = bool(body.get("webSearch", False)) and (not use_second_speaker or two_player_mode)
            character_prompt = str(body.get("systemPrompt") or "").strip()
            user_address = str(body.get("userAddress") or "あなた").strip() or "あなた"
            tts_caption = str(body.get("ttsCaption") or IRODORI_CAPTION).strip()
            style_guide = str(body.get("styleGuide") or "").strip()
            cfg_scale_text = sanitize_cfg_scale(body.get("cfgScaleText"), IRODORI_CFG_SCALE_TEXT)
            cfg_scale_caption = sanitize_cfg_scale(body.get("cfgScaleCaption"), IRODORI_CFG_SCALE_CAPTION)
            cfg_scale_speaker = sanitize_cfg_scale(body.get("cfgScaleSpeaker"), IRODORI_CFG_SCALE_SPEAKER)
            reference_path = sanitize_reference_path(body.get("referencePath"), IRODORI_REF_WAV)
            second_reference_path = sanitize_reference_path(body.get("secondReferencePath"), LUVIA_REF_WAV)
            tts_backend_mode = str(body.get("ttsBackendMode") or "local").strip().lower()
            second_tts_host = str(body.get("secondTtsHost") or body.get("secondTtsUrl") or "").strip()
            second_tts_url = normalize_remote_tts_url(second_tts_host)
            context_limit = int(body.get("contextLimit") or DEFAULT_CONTEXT_LIMIT)
            existing_web_context = str(body.get("webContext") or "").strip()
            web_topic = str(body.get("webTopic") or "").strip()
            raw_messages = [
                item
                for item in history
                if isinstance(item, dict) and item.get("role") in {"user", "assistant"}
            ]
            search_results: list[dict[str, str]] = []
            web_query = ""
            web_context = existing_web_context
            if use_web_search:
                web_query_source = web_topic or user_text
                web_query = build_continuous_web_query(web_query_source, raw_messages)
                try:
                    search_results = web_search(web_query, limit=3)
                except Exception as exc:
                    search_results = [{"title": "検索エラー", "url": "", "snippet": str(exc)}]
                web_context = format_web_results(web_query, search_results)
            if web_context:
                raw_messages.append(
                    {
                        "role": "user",
                            "content": (
                                f"今回の進行指示/質問: {compact_text(web_topic or user_text, 180)}\n"
                                f"共有Webメモとして保持中の検索語: {web_query or '前回のお題'}\n"
                                f"{web_context}"
                            ),
                    }
                )
            raw_messages.append({"role": "user", "content": user_text})
            messages, context_stats = compact_messages_for_context(raw_messages, context_limit)
            # RAG: 直近文脈（compact_messages_for_context）はそのまま活かしつつ、
            # 意味的に近い過去ログを裏で top-k 抽出して参考枠として差し込む（併用）。
            # 失敗時・ヒット0件時は memory_block を空にして従来動作へフォールバックする。
            memory_block = ""
            memory_mode = ""
            # 列挙・網羅の質問かどうか。要約で枠を空ける際に「件数を減らさない」よう
            # 指示を変える必要があるので、request_lmstudio まで持っていく。
            memory_enum = False
            if rag_memory is not None:
                try:
                    # 生発話は会話的な雑音（挨拶・枕詞）が多く意味検索の精度が落ちるため、
                    # LLM で焦点を絞った検索クエリへ書き換えてから recall する。直近の
                    # assistant 発話を背景に渡し、「他には？」等の follow-up も解決させる。
                    # 失敗・空なら原文へフォールバック（RAG は純粋な追加レイヤー）。
                    recent_ctx = ""
                    for _m in reversed(messages):
                        if _m.get("role") == "assistant":
                            recent_ctx = compact_text(_m.get("content"), 200)
                            break
                    recall_queries, intent_hint = rewrite_recall_queries(
                        user_text,
                        recent_ctx,
                        model=model,
                        generation_mode=generation_mode,
                        user_name=user_address,
                        char_name=speaker,
                    )
                    recall_queries = recall_queries or [user_text]
                    if recall_queries != [user_text]:
                        print(
                            "[rag] recall queries rewritten -> "
                            + " | ".join(compact_text(q, 60) for q in recall_queries)
                        )
                    # 時系列・列挙・期間の意図を検出する。正規表現を主とし、LLM が返した
                    # タグ（intent_hint）は正規表現が何も拾えなかったときだけ採用する
                    # （判定のブレを最小化するため。外しても従来のスコア順想起に落ちるだけ）。
                    recall_intent = detect_recall_intent(user_text)
                    if intent_hint:
                        tags = set(intent_hint.split(","))
                        if not recall_intent["temporal"]:
                            for tag in ("first", "last", "when"):
                                if tag in tags:
                                    recall_intent["temporal"] = tag
                                    break
                        if "enum" in tags:
                            recall_intent["enum"] = True
                    if recall_intent["temporal"] or recall_intent["enum"] or recall_intent["period"]:
                        print(
                            "[rag] intent temporal="
                            f"{recall_intent['temporal'] or '-'} enum={recall_intent['enum']}"
                            f" period={recall_intent['period'] or '-'}"
                            f" (llm hint={intent_hint or '-'})"
                        )
                    # dedup は「LLM が実際に見る圧縮後 messages」を基準にする。
                    # raw_messages（全履歴）基準にすると、文脈から溢れて要約に畳まれた
                    # 古い記憶まで除外され、RAG が本来補うべき“文脈落ちした過去”を差し
                    # 込めなくなる（＝想起の盲点・幻覚の原因）。
                    recent_user_texts = [
                        item.get("content")
                        for item in messages
                        if item.get("role") == "user"
                    ]
                    # 2人だけモードのお題と 1P 通常会話の記憶が混ざらないよう、現在の
                    # ターンと同じ会話モードの記憶だけを想起対象にする。
                    recall_mode = "two_only" if two_only_mode else "normal"
                    # ベクトル（意味）・語彙（全件一致）・台帳（集計）の 3 チャネルで想起し、
                    # 意図に応じて年表／一覧へ整形する。詳細は recall_for_turn を参照。
                    recall_result = recall_for_turn(
                        character_id,
                        queries=recall_queries,
                        slot=speaker_slot,
                        mode=recall_mode,
                        intent=recall_intent,
                        recent_user_texts=recent_user_texts,
                        user_name=user_address,
                        char_name=speaker,
                    )
                    memory_block = recall_result["block"]
                    memory_mode = recall_result["mode"]
                    memory_enum = bool(recall_intent.get("enum"))
                    # 想起が「どのチャネルから何件を」差し込んだかを可視化（想起漏れ・
                    # 誤想起の切り分け用）。実発話が曖昧（「作った料理」＝誰が作った？）だと
                    # 別の記憶を拾って回答がすり替わるため、件数と内訳を残す。
                    _stats = recall_result["stats"]
                    if memory_block:
                        print(
                            f"[rag] recalled shown={_stats.get('shown', 0)} "
                            f"(vector={_stats.get('vector', 0)} "
                            f"lexical={_stats.get('lexical', 0)} "
                            f"ledger={_stats.get('ledger', 0)}) "
                            f"mode={memory_mode or 'plain'} "
                            f"filters={_stats.get('filters') or '-'}"
                        )
                    else:
                        print("[rag] recalled 0 memories")
                except Exception as exc:
                    print(f"[rag] recall failed: {type(exc).__name__}: {exc}")
                    memory_block = ""
                    memory_mode = ""
                    memory_enum = False
            reply, model_used, llm_emoji, chunk_limit, segments = request_lmstudio(
                messages,
                model,
                auto_emoji=auto_emoji,
                reply_length=reply_length,
                character_prompt=character_prompt,
                user_address=user_address,
                no_dialogue=no_dialogue,
                speaker=speaker,
                two_only_mode=two_only_mode,
                style_guide=style_guide,
                generation_mode=generation_mode,
                memory_block=memory_block,
                memory_mode=memory_mode,
                memory_enum=memory_enum,
            )
            # 合成部分（感情セグメント→TTS→結合）は /api/regenerate と共通の関数へ委譲する。
            render = render_reply_audio(
                reply,
                segments,
                speaker_slot=speaker_slot,
                tts_caption=tts_caption,
                steps=steps,
                speech_rate=speech_rate,
                emoji_style=emoji_style,
                fallback_emoji=llm_emoji,
                chunk_limit=chunk_limit,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_caption=cfg_scale_caption,
                cfg_scale_speaker=cfg_scale_speaker,
                reference_path=reference_path,
                second_reference_path=second_reference_path,
                tts_backend_mode=tts_backend_mode,
                second_tts_url=second_tts_url,
            )
            audios = render["audios"]
            chunks = render["chunks"]
            seg_meta = render["segMeta"]
            combined_audio = render["combined"]
            representative_emoji = render["representativeEmoji"]
            effective_emoji = render["effectiveEmoji"]
            duration_scale = render["durationScale"]
            use_remote_tts = render["useRemoteTts"]
            reference_wav = render["referenceWav"]
            append_chat_log(
                {
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "user": user_text,
                    "reply": reply,
                    "speaker": speaker,
                    "model": model_used,
                    "replyLength": reply_length,
                    "speechRate": speech_rate,
                    "durationScale": duration_scale,
                    "emojiStyle": effective_emoji,
                    "llmEmojiStyle": llm_emoji,
                    "autoEmoji": auto_emoji,
                    "noDialogue": no_dialogue,
                    "webSearch": use_web_search,
                    "twoOnlyMode": two_only_mode,
                    "ttsBackendMode": tts_backend_mode,
                    "secondTtsUrl": second_tts_url,
                    "secondTtsRemote": use_remote_tts,
                    "webQuery": web_query,
                    "webContext": web_context,
                    "webResults": search_results,
                    "ttsCaption": tts_caption,
                    "userAddress": user_address,
                    "reference": str(reference_wav),
                    "characterPrompt": character_prompt,
                    "expression": expression_for_emoji(representative_emoji),
                    "segments": seg_meta,
                    "chunkCount": len(chunks),
                    "chunks": chunks,
                    "combinedUrl": (combined_audio or {}).get("url"),
                    "audios": [
                        {
                            "text": item.get("text"),
                            "ttsText": item.get("ttsText"),
                            "emojiStyle": item.get("emojiStyle"),
                            "speechRate": item.get("speechRate"),
                            "durationScale": item.get("durationScale"),
                            "expression": item.get("expression"),
                            "elapsed": item.get("elapsed"),
                            "url": item.get("url"),
                        }
                        for item in audios
                    ],
                    "contextStats": context_stats,
                }
            )
            # 感情キャプション付きの返答テキストを専用履歴に記録する。
            append_emotion_log(
                {
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "user": user_text,
                    "speaker": speaker,
                    "speakerSlot": speaker_slot,
                    "model": model_used,
                    "reply": reply,
                    "annotatedReply": build_annotated_reply(reply, seg_meta),
                    "segments": seg_meta,
                }
            )
            # RAG: 今回の 1 往復（ユーザー発言＋返答）を長期記憶へベクトル保存する。
            # 失敗しても本処理は継続（純粋な追加レイヤーとして扱う）。
            if rag_memory is not None and reply:
                try:
                    # 2人だけモードは user_text がお題/進行指示なので mode で区別し、
                    # 想起時に「ユーザー」ではなく「お題＋話者名」で差し込ませる。
                    turn_mode = "two_only" if two_only_mode else "normal"
                    source_id = rag_memory.save_turn(
                        character_id,
                        speaker_slot,
                        user_text,
                        reply,
                        mode=turn_mode,
                        speaker=speaker,
                    )
                    # 事実台帳への増分抽出。LLM 抽出を含みうるので必ず別スレッドへ逃がす
                    # （返答の応答時間に影響させない）。失敗しても会話には無影響。
                    if source_id and _LEDGER_LIVE and fact_extract is not None:
                        threading.Thread(
                            target=extract_facts_for_turn,
                            args=(character_id, source_id),
                            kwargs={
                                "user_text": user_text,
                                "reply_text": reply,
                                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                "slot": speaker_slot,
                                "mode": turn_mode,
                                "speaker": speaker,
                                "user_name": user_address,
                                "char_name": speaker,
                                "model": model,
                                "generation_mode": generation_mode,
                                "use_llm": _LEDGER_LIVE_LLM,
                            },
                            daemon=True,
                        ).start()
                except Exception:
                    pass
            self.send_json(
                200,
                {
                    "reply": reply,
                    "speaker": speaker,
                    "model": model_used,
                    "chunks": chunks,
                    "emojiStyle": representative_emoji,
                    "expression": expression_for_emoji(representative_emoji),
                    "segments": seg_meta,
                    "llmEmojiStyle": llm_emoji,
                    "autoEmoji": auto_emoji,
                    "replyLength": reply_length,
                    "speechRate": speech_rate,
                    "durationScale": duration_scale,
                    "audios": audios,
                    "combined": combined_audio,
                    "contextStats": context_stats,
                    "webSearch": use_web_search,
                    "twoOnlyMode": two_only_mode,
                    "ttsBackendMode": tts_backend_mode,
                    "secondTtsUrl": second_tts_url,
                    "secondTtsRemote": use_remote_tts,
                    "webQuery": web_query,
                    "webContext": web_context,
                    "webResults": search_results,
                },
            )
        except urllib.error.URLError as exc:
            self.send_json(502, {"error": f"LM Studio request failed: {exc}"})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[server] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    host = os.environ.get("CHAT_HOST", "127.0.0.1")
    port = int(os.environ.get("CHAT_PORT", "7862"))
    STATIC_ROOT.mkdir(parents=True, exist_ok=True)
    (STATIC_ROOT / "generated").mkdir(parents=True, exist_ok=True)
    CHARACTER_ROOT.mkdir(parents=True, exist_ok=True)
    run_startup_migrations()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Irodori LM Studio chat: http://{host}:{port}/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
