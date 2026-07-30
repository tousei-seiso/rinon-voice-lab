# Rinon Voice Lab — Tousei Edition

Local character chat and speech app. Windows is the primary tested platform,
and macOS support is experimental.

日本語版: [README.ja.md](README.ja.md)

Rinon Voice Lab connects:

- LM Studio OpenAI-compatible local chat
- Irodori-TTS VoiceDesign speech generation
- Editable 1P/2P character profiles
- Character portraits and expression variants
- Optional lightweight Web-search notes for LLM prompts
- Optional 2P remote TTS on a second PC

## Improvements over upstream

This is a fork of [sakugetu/rinon-voice-lab](https://github.com/sakugetu/rinon-voice-lab),
focused on higher-quality Japanese character speech and day-to-day usability.

- **Emotion-aware acting** — per-emotion caption segmentation, one-utterance-per-segment
  synthesis (fewer short-chunk artifacts), single/sustained vocal-effect emoji handling,
  and per-character style guides & Num Steps to stop timbre drift.
- **Stable timbre** — per-character CFG Scale (text/caption/speaker), a raised default
  speaker CFG, and one fixed seed shared across all chunks of a reply.
- **Better Japanese TTS** — English-to-kana normalization to prevent runaway output,
  multiple kana dictionaries with an alkana fetch helper, and utf-8-sig (BOM) CSV loading.
- **Robust LM Studio integration** — configurable timeout (300s), hardened JSON parsing,
  json_schema structured output with safe fallback, assistant-prefill to avoid empty
  replies, and selectable generation modes (prefill/original/quality_guard/unlimited).
- **Single-file audio** — split Irodori output combined into one WAV, including correct
  handling of IEEE-float (format 3) WAV via manual RIFF parsing.
- **Usability** — selectable replies with targeted play/download/save, persisted
  annotations across reloads, per-reply regenerate/delete, and per-character logs,
  saved audio, and context limits.
- **Local long-term memory (RAG)** — semantically retrieves the most relevant past
  conversations for the current message and injects them as a reference block, so
  characters stay consistent over long histories. Embeddings run fully on CPU (no VRAM)
  via `intfloat/multilingual-e5-small` with a dual backend (fastembed/ONNX, or
  transformers+torch), stored per-character in SQLite. Existing logs/histories are left
  untouched and it degrades gracefully when the optional deps are absent. A one-shot
  backfill script (`tools/backfill_rag_memory.py`) imports existing histories.
- **Sharper recall** — retrieval is filtered by conversation mode (1P vs 2P) and speaker
  slot so 2P topics don't leak into 1P chats; the raw message is rewritten by the LLM into
  a focused search query (drops filler and negated/excluded terms, keeps the action's
  subject/object, and falls back to the user as subject when it is omitted); enumeration
  questions ("list them all") fan out into several queries whose results are merged, and
  are listed exhaustively only when explicitly asked; and near-duplicate collapse plus
  `top_k`/`min_score` tuning keep context-dropped memories in reach. Rebuild/diagnose
  helpers: `tools/rebuild_from_chatlog.py`, `tools/diagnose_recall.py`.
- **Three-channel recall (time, coverage, who-did-what)** — cosine top-k structurally
  cannot answer some questions: "what was the *first* book you bought me?" ignores time,
  and "list *every* dish I cooked" overflows the top-k window as soon as there are more
  matches than slots. Three complementary channels run together:
  - *Vector* — the usual semantic search for topically close memories.
  - *Lexical (unbounded)* — SQLite FTS5(trigram) plus `LIKE`, matching by occurrence
    rather than rank so nothing is lost to a result-count window. Terms of 3+ characters
    go to FTS5 and 1–2 character terms (e.g. the single kanji 本) to `LIKE`, since trigram
    never matches shorter queries. Japanese inflection is folded to stems (買った → 買)
    so a query never misses a differently-conjugated body text.
  - *Fact ledger (aggregation)* — each turn is distilled into who / to whom / did what /
    to what in a `facts` table, so enumeration becomes `SELECT DISTINCT` and returns
    **every** row. The action's **direction** (`user->char` / `char->user`) is stored
    structurally, which is what keeps "the dish I cooked for you" from being confused with
    "the dish you cooked for me". Extraction is hybrid: Japanese benefactive forms
    (〜してあげた = speaker→other, 〜してくれた = other→speaker) combined with the
    message role settle most turns with no LLM call, and only the undecided ones go to the
    local LLM. Unresolved fields are kept as "direction unknown" rather than dropped, and
    the ledger is always presented alongside the **source turns** so the original text
    remains the ground truth. Only facts that help recall what happened are recorded —
    validated against real logs, the ledger rejects facts with no object, the verbs *say*
    and *see* (the conversation log is already the record of what was said, and these were
    the single largest source of junk objects), contentless objects (particles, formal
    nouns, pronouns, and words like greeting / feeling / face), and it downgrades an
    action's direction to unknown when a third party is involved, so filtering for "what I
    did for you" never pulls in acts aimed at someone else. It also rejects subjects that
    are clause fragments rather than names (が marks a subject but also lives inside verbs
    like 立ち上がって), strips continuative clauses and time words off objects, ignores
    benefactives separated by another case particle, and drops rule-derived rows on turns
    the LLM already read. Asking "what did *I* do" also pulls in that person's own actions
    (`direction='self'`), which a direction match alone cannot reach. Measured over 771 turns, ~87%
    of turns need the LLM pass, so build the ledger with LM Studio running rather than
    `--rule-only`.
    Alongside direction, every fact carries its **modality** and its **event time**, because
    Japanese verbs keep the same stem while only tense and mood change — read by surface
    pattern alone, "next time I'll cook you a udon carbonara" was recorded as a dish that
    *was* cooked, and "we went to the Tamagawa fireworks a year ago" as something done on
    the day it was *mentioned*. So `modality` (`done` / `plan` / `wish` / `negated` /
    `unknown`) is decided from the tail of that specific verb — non-past benefactives
    (〜てあげる) are offers, not completions — and `occurred` holds the event's own time
    resolved from expressions like 1年前 / 去年の夏 / 明日 against the turn's timestamp
    (`ts` is only the day it was talked about). An expression with no year (8月5日, 秋)
    resolves *backwards* for a recollection but *forwards* for a plan — otherwise next
    week's plan lands a year in the past. Enumeration defaults to `done` (plus
    `unknown`, so a missed call never drops a real fact), plans are shown in their own
    section rather than discarded, and period filters match on either time so a
    reminiscence is not lost. `tools/check_fact_modality.py` pins these judgements down
    with example sentences.
- **Timestamp-aware recall** — asks like "the very first", "the last / lately", "when",
  "last summer", "in March", "two years ago" are detected by regex (with the rewrite LLM's
  intent tag as a fallback), a wide candidate pool is gathered and then re-sorted by time.
  The prompt receives a **chronological timeline** with dates and pre-computed elapsed time
  ("about 1 year 7 months ago" — the server does the arithmetic, never the LLM), plus the
  span of what is on record so "no record before this" is not mistaken for "it never
  happened". Period expressions resolve to `since`/`until` filters.
  Tools: `tools/build_fact_ledger.py` (bulk ledger build; `--rule-only` needs no LLM,
  resumes where it stopped; `--list --modality plan` reviews what was filed as a plan,
  `--list --object <word>` looks a specific item up, and `--fix-occurred` recomputes event
  times from the stored `time_hint` without any LLM call),
  `tools/check_fact_modality.py` (regression check for the tense / event-time rules, no LLM
  or DB needed), `tools/diagnose_temporal.py` (inspect which channel a
  temporal/enumeration ask falls through, no LLM needed), `tools/audit_memory.py`
  (reconcile `chat.jsonl` / `history.json` / `memories` counts to separate *missing saves*
  — which no retrieval change can fix — from retrieval misses).
- **Editing history stays in sync** — the log files form a hierarchy: `logs/chat.jsonl` is
  the root source of truth, and `history.json`, `memory.sqlite3` and `chat_emotion.jsonl`
  are derived from it. `tools/sync_memory.py` reconciles the downstream files
  **differentially** (no re-embedding of the whole history, and the ledger survives):
  removed turns are deleted along with the facts extracted from them, missing turns are
  added, edited timestamps propagate to the ledger, and facts whose source turn is gone are
  pruned. Matching is by user text + reply + conversation mode (`ts` is excluded since it
  may be hand-edited).
  `--source chatlog` (default) treats `chat.jsonl` as the source and regenerates
  `history.json`, syncs the database, and rebuilds `chat_emotion.jsonl` (`--sync-emotion`).
  Emotion captions come from `chat.jsonl` alone: every turn carries `segments`
  (`{style, emoji, text}`), so the annotated display text is reassembled with
  `emotion_caption.build_annotated_reply` — `chat_emotion.jsonl`'s `annotatedReply` is that
  function's output, not an independent source. The handful of turns from the day segment
  captions shipped lack `segments[].text`; `tools/repair_chatlog_segments.py` restores those
  from the recorded `chunks`/`audios`.
  `--source history --propagate` covers the reverse direction, pushing deletions back into
  `chat.jsonl` and `chat_emotion.jsonl` — without that, a conversation removed only
  downstream reappears the next time a full rebuild runs. A full `--reset` rebuild deletes
  the database file and therefore the ledger too; `rebuild_rag_from_history.py --extract`
  rebuilds both.
  The `modality` / `occurred` columns are added to an existing ledger automatically, but rows
  already stored as "done" stay that way — re-extract to fix them:
  `build_fact_ledger.py --redo-rule` (seconds, no LLM, rule-derived facts only) and, if the
  LLM-derived ones matter too, `--redo-verb 作る` or `--redo --since <date>` with LM Studio
  running.
- **Deleting in the UI removes the whole turn everywhere** — your message and the reply are
  dropped from all four places at once (`history.json`, `chat.jsonl`,
  `chat_emotion.jsonl`, and `memory.sqlite3` including the facts extracted from that turn),
  with a `.bak` written before the raw logs are rewritten. A prompt shared by several 2P
  replies is kept while any reply still references it, and generated audio files are left
  alone since other entries may point at them. This runs through a dedicated
  `/api/delete-turn` rather than the session auto-save: auto-save only writes the *current*
  conversation context, so inferring deletions from it could not be told apart from Clear
  Context and would wipe raw logs that must be kept — `history.json` and `chat.jsonl` are
  expected to differ.
- **History foundation** — session history auto-saves on every turn (conversation mode,
  speaker, and timestamp preserved), and speakers are keyed by slot (1P=main / 2P=second)
  so same-named 1P/2P characters never get mixed up in logs or recall.

## Screenshots / 画面モード

### 1P Mode / 1Pモード

1Pモードは、1人のキャラクターと会話しながら、LM Studio の応答を
Irodori-TTS で読み上げる基本モードです。キャラ設定、TTS Caption、
Web検索、話速、感情スタイルを同じ画面で調整できます。

![Rinon Voice Lab 1P mode](docs/images/rinon-1p-mode.png)

### 2P Mode / 2Pキャラモード

2Pキャラモードでは、1Pと2Pのキャラクターを同じ画面に表示し、
二人の会話を交互に進められます。2人だけで話すモード、2P音声の別PC生成、
キャラクターごとの設定やTTS Captionにも対応しています。

![Rinon Voice Lab 2P mode](docs/images/rinon-2p-mode.png)

## Support / サポートについて

This is a personal experimental release. Please do not expect support,
maintenance, compatibility guarantees, or help with individual environments.
Use it as a reference implementation or a local experiment.

個人の実験的な公開物です。サポート、継続メンテナンス、環境ごとの動作保証、
個別の導入支援は期待しないでください。参考実装またはローカル実験用として
利用してください。

The app is designed to run from any install folder. It does not require a fixed
drive such as `H:`. By default, Irodori-TTS is installed next to this app:

```text
SomeFolder\
  RinonVoiceLab\
  Irodori-TTS\
```

## Requirements

- Windows 10/11
- macOS 14 or newer on Apple Silicon (experimental)
- Python 3.10 or newer
- Git
- LM Studio with the local server enabled
- A local chat model loaded in LM Studio
- NVIDIA GPU strongly recommended for Irodori-TTS
- `uv` for Irodori-TTS dependency setup

The Rinon Voice Lab wrapper uses only the Python standard library directly.
`requirements.txt` intentionally contains no app-level packages. Irodori-TTS is
installed into its own virtual environment by `tools\install_irodori_tts.ps1`.

macOS cannot use CUDA. On Apple Silicon, Irodori-TTS can use PyTorch MPS when it
is available; otherwise it falls back to CPU. MPS/CPU use `fp32`, because
Irodori-TTS `bf16` inference is CUDA/XPU-only. Voice generation may be much
slower than on an NVIDIA GPU.

## Quick Start (Windows)

1. Clone or download this repository.
2. Start LM Studio and enable the OpenAI-compatible local server.
3. Load a chat model, for example `gemma-4-12b-it`.
4. Double-click `start_chat_uv.bat`.
5. Open `http://127.0.0.1:7862/`.

If Irodori-TTS is not installed yet, `start_chat_uv.bat` runs
`tools\install_irodori_tts.ps1` automatically. The first install can take a long
time because PyTorch and model dependencies are large.

## Quick Start (macOS)

1. Start LM Studio and enable the OpenAI-compatible local server.
2. Load a chat model.
3. Open Terminal in this repository.
4. Run:

```bash
chmod +x start_chat_mac.sh tools/install_irodori_tts.sh
./start_chat_mac.sh
```

5. Open `http://127.0.0.1:7862/`.

If Irodori-TTS is not installed yet, `start_chat_mac.sh` runs
`tools/install_irodori_tts.sh`. On macOS, the installer uses
`uv sync --extra cpu`. That extra falls back to standard PyPI PyTorch wheels on
macOS, so Apple Silicon can use MPS when PyTorch reports it as available.

The macOS script uses Python 3.10 by default because PyTorch wheels may not be
available for newer Python versions such as 3.14. To override it:

```bash
IRODORI_PYTHON_VERSION=3.13 ./start_chat_mac.sh
```

To place Irodori-TTS somewhere else:

```bash
IRODORI_ROOT="$PWD/.deps/Irodori-TTS" ./start_chat_mac.sh
```

## Manual Install

Run this from the app folder:

```powershell
powershell -ExecutionPolicy Bypass -File tools\install_irodori_tts.ps1
```

The installer defaults to CUDA 12.8 wheels:

```powershell
uv sync --extra cu128
```

For CPU-only setup:

```powershell
powershell -ExecutionPolicy Bypass -File tools\install_irodori_tts.ps1 -TorchExtra cpu
```

CPU mode is mainly for testing. Voice generation can be very slow.

For macOS:

```bash
IRODORI_TORCH_EXTRA=cpu tools/install_irodori_tts.sh
```

## Configuration

Useful environment variables:

| Variable | Default | Purpose | Upstream | Tousei |
| --- | --- | --- | :-: | :-: |
| `IRODORI_ROOT` | `..\Irodori-TTS` next to this app | Irodori-TTS checkout and virtual environment | ✅ | ✅ |
| `LM_STUDIO_URL` | `http://127.0.0.1:1234/v1` | LM Studio OpenAI-compatible endpoint | ✅ | ✅ |
| `LM_STUDIO_MODEL` | `gemma-4-12b-it` | Preferred model name | ✅ | ✅ |
| `LM_STUDIO_CONTEXT_LIMIT` | `8200` | Visible context budget | ✅ | ✅ |
| `LM_SERIALIZE_REQUESTS` | `1` | Send one request at a time so the server never opens a second slot (`0` allows overlap) | — | ✅ |
| `LM_CACHE_PROMPT` | `aux` | Where to send `cache_prompt=false`: `aux` (helper generations only), `all`, or `off` | — | ✅ |
| `LM_KV_CACHE_RELEASE` | `idle` | Explicit KV-cache release: `idle` (once traffic settles), `each` (every request), or `off` | — | ✅ |
| `LM_KV_CACHE_RELEASE_DELAY` | `2` | Seconds to wait before an `idle` release; raise it to keep the cache warm across quick turns | — | ✅ |
| `LM_KV_CACHE_RELEASE_TIMEOUT` | `5` | Timeout for the release API (seconds) | — | ✅ |
| `VRAM_RELEASE_TORCH` | `1` | Also hand this process's torch CUDA cache (Irodori-TTS) back via `empty_cache()` | — | ✅ |
| `VRAM_MEMORY_LOG` | `1` | Log the GPU breakdown (total plus this process's torch reservation) around each sweep | — | ✅ |
| `IRODORI_TORCH_EXTRA` | `cu128` | Installer torch extra: `cu128`, `cpu`, `rocm`, or `xpu` | ✅ | ✅ |
| `IRODORI_MODEL_DEVICE` | `auto` | Irodori-TTS model device: `auto`, `cuda`, `mps`, `cpu`, or `xpu` | ✅ | ✅ |
| `IRODORI_MODEL_PRECISION` | `auto` | Model precision: `auto`, `fp32`, or `bf16` | ✅ | ✅ |
| `IRODORI_CODEC_DEVICE` | `auto` | Codec device, usually the same as the model device | ✅ | ✅ |
| `IRODORI_CODEC_PRECISION` | `auto` | Codec precision. macOS uses `fp32` | ✅ | ✅ |
| `RAG_MEMORY_ENABLED` | `1` | Enable RAG long-term memory (`0` to disable) | — | ✅ |
| `RAG_EMBED_MODEL` | `intfloat/multilingual-e5-small` | Embedding model | — | ✅ |
| `RAG_RECALL_TOP_K` | `16` | Max memories recalled per message | — | ✅ |
| `RAG_RECALL_MIN_SCORE` | `0.75` | Recall similarity threshold | — | ✅ |
| `RAG_RECALL_DEDUP` | `1` | Collapse near-duplicate memories (`0` to disable) | — | ✅ |
| `RAG_QUERY_REWRITE` | `1` | LLM rewrite of the recall query (`0` uses the raw text) | — | ✅ |
| `RAG_QUERY_REWRITE_MODE` | empty | Generation mode for the rewrite (empty follows the chat; `prefill` is faster) | — | ✅ |
| `RAG_QUERY_REWRITE_MULTI` | `3` | Max recall queries generated for enumeration asks (`1` = single) | — | ✅ |
| `RAG_LEXICAL_ENABLED` | `1` | Enable the lexical channel (FTS5+LIKE); `0` = vector only | — | ✅ |
| `RAG_LEXICAL_LIMIT` | `24` | Max rows taken from the lexical channel | — | ✅ |
| `RAG_LEXICAL_LIMIT_ENUM` | `48` | Lexical limit for enumeration asks | — | ✅ |
| `RAG_LEXICAL_SLACK` | `0.03` | Similarity slack for lexical hits (independent evidence) | — | ✅ |
| `RAG_TEMPORAL_POOL_K` | `64` | Candidate pool gathered before re-sorting by time | — | ✅ |
| `RAG_TEMPORAL_K` | `8` | Timeline rows injected into the prompt | — | ✅ |
| `RAG_TEMPORAL_BAND` | `0.02` | Score band treated as "on topic" for temporal selection (measured: e5 scores collapse into 0.80–0.84, so widening it lets an unrelated old memory claim "the first") | — | ✅ |
| `RAG_LEDGER_ENABLED` | `1` | Enable the fact ledger (`0` disables read/write) | — | ✅ |
| `RAG_LEDGER_LIMIT` | `60` | Max facts injected from the ledger | — | ✅ |
| `RAG_LEDGER_TURNS` | `8` | Source turns included as evidence for the ledger | — | ✅ |
| `RAG_LEDGER_PLANS` | `12` | Plans/wishes listed as explicit counter-evidence when asked what was actually done (`0` disables) | — | ✅ |
| `RAG_LEDGER_ALWAYS` | `0` | Consult the ledger even for non-temporal/non-enumeration asks | — | ✅ |
| `RAG_LEDGER_LIVE` | `1` | Extract the current turn into the ledger after replying (background) | — | ✅ |
| `RAG_LEDGER_LIVE_LLM` | `1` | Use the LLM for live extraction (`0` = rule-only, free) | — | ✅ |
| `RAG_FACT_EXTRACT_MAXTOK` | `256` | Token cap for the fact-extraction LLM | — | ✅ |

## Character Data

Characters live under `Character\<character-id>\`.

Each character folder can contain:

- `profile.txt` for hand editing
- `profile.json` for structured save/load
- `reference\` for voice reference audio
- `expressions\<slot>\` for expression images

Use the Options dialog in the app to edit character names, prompts, TTS
captions, reference audio, and expression images.

## Optional 2P Remote TTS

By default, both 1P and 2P voices are generated on the local Irodori-TTS
environment.

In the main toolbar, use `TTS PC` to choose the runtime mode:

- `1 PC`: generate both 1P and 2P voices on this machine.
- `2 PCs`: generate 1P locally and send only 2P voice generation to a second
  machine.

When `2 PCs` is selected, enter the second machine in `2P IP`. An IP-only value
such as `192.168.0.10` is expanded to `http://192.168.0.10:7874`. You can also
enter `192.168.0.10:7874` or a full URL.

On the second Windows machine, start the lightweight remote TTS server:

```powershell
$env:IRODORI_ROOT = "H:\AI\Irodori-TTS"
$env:LUVIA_SERVER_PORT = "7874"
python tools\remote_luvia_tts_server.py
```

The second machine must have Irodori-TTS installed and reachable from the main
machine. The remote server exposes `/health` and `/synthesize`.

On macOS or Linux, start the remote TTS server with:

```bash
IRODORI_ROOT="$PWD/../Irodori-TTS" \
LUVIA_SERVER_PORT=7874 \
IRODORI_MODEL_DEVICE=auto \
python tools/remote_luvia_tts_server.py
```

## External Speak Mode

Rinon Voice Lab can receive short text from Codex, Claude Code, or another
local tool and speak it through the open character UI.

Start Rinon Voice Lab, open `http://127.0.0.1:7862/`, then POST UTF-8 JSON:

```powershell
$body = @{
  text = "リノンから外部スピークのテストだよ。"
  emoji = "🤭"
  caption = "soft cheerful Japanese anime voice, clear pronunciation"
  speakerSlot = "main"
  steps = 8
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  http://127.0.0.1:7862/api/speak `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

Common payload keys:

| Key | Purpose |
| --- | --- |
| `text` | Text to speak |
| `emoji` / `emojiStyle` | Irodori style emoji |
| `caption` / `ttsCaption` | VoiceDesign acting caption |
| `speakerSlot` | `main` or `second` |
| `referencePath` | Optional reference wav path |
| `steps` | Irodori generation steps |
| `speechRate` | `normal` or `fast` |

The browser polls `/api/speak-events` and plays new events with the normal
character animation, expression switching, panning, and audio save controls.

## Runtime Files

These local runtime files are ignored by Git and should be removed before
distributing a ZIP copy:

- `logs/`
- `profiles/`
- `saved_audio/`
- `static/generated/`
- Python caches and virtual environments

## Validation

Useful development checks:

```powershell
node --check static\app.js
$env:PYTHONDONTWRITEBYTECODE='1'
..\Irodori-TTS\.venv\Scripts\python.exe -B -m py_compile app.py tools\remote_luvia_tts_server.py
```

macOS:

```bash
node --check static/app.js
PYTHONDONTWRITEBYTECODE=1 python3.10 -B -m py_compile app.py tools/remote_luvia_tts_server.py
```

## License

MIT License. See [LICENSE](LICENSE).
