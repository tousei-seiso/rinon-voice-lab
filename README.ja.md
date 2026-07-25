# Rinon Voice Lab 日本語README

Rinon Voice Lab は、LM Studio のローカルLLMと Irodori-TTS をつないで、キャラクター会話と音声読み上げを行うローカルアプリです。主な確認環境は Windows ですが、macOS では実験的に起動できるようにしています。

主な機能:

- LM Studio の OpenAI互換ローカルAPIと連携
- Irodori-TTS VoiceDesign による日本語音声生成
- 1P/2P キャラクターの設定、名前、TTS Caption、表情画像を編集
- 会話ログ、セッション、キャラクターデータの保存と読み込み
- 簡易Web検索メモをLLMプロンプトへ追加
- 2P音声だけを別PCの Irodori-TTS へ送るリモートTTSモード

## 本家からの主な改良点

このリポジトリは本家 [sakugetu/rinon-voice-lab](https://github.com/sakugetu/rinon-voice-lab) を
ベースに、日本語キャラクター音声の品質と実運用での使い勝手を重点的に強化したフォークです。

### 1. 感情キャプションによる演技表現の強化
- LM の返答を感情ごとにセグメント分割し、区間ごとに Irodori VoiceDesign の
  演技キャプションを付与（棒読み回避）
- 各感情セグメントを1発話として生成し、短チャンク由来の音切れ・ノイズを抑制
- 効果系絵文字を「単発／持続」に分類して発話位置へ適切に反映
- キャラクターごとの Style Guide で、感情キャプションによる声質のブレを抑制
- キャラクターごとの Num Steps 設定（既定40）
- 感情注釈つき返答を `logs/chat_emotion.jsonl` へ記録

### 2. 声質の安定化（CFG Scale / シード制御）
- CFG Scale（text / caption / speaker）をキャラクターごとに編集可能化
- 未調整キャラの声質を安定させるため speaker CFG の既定値を引き上げ
- 1返答内の全文チャンクで共通シードを使用し、声質のばらつきを解消
- Irodori へ渡した CFG Scale 値をログ出力し、キャラ別に検証可能に

### 3. 日本語TTS品質の底上げ（かな辞書・英語正規化）
- 英単語→かな正規化で、英語混じり文による Irodori の暴走（runaway）を防止
- 複数のかな辞書に対応、`alkana` 辞書取得ヘルパーを追加
- 辞書名をプロジェクトルート基準でも解決
- 辞書は「訳語」ではなく「読み」を使うようガイダンスを修正
- Excel由来の BOM 付き CSV を utf-8-sig で正しく読み込み

### 4. LM Studio 連携の堅牢化
- チャットのタイムアウトを設定可能に（既定300秒）
- 崩れた出力に強い JSON パースへ改善
- structured output（json_schema）に対応し、失敗時は安全にフォールバック
- assistant プレフィルで reasoning を抑制し、空応答を回避
- 生成モード設定を追加（prefill / original / quality_guard / unlimited）

### 5. 分割音声の1本化
- Irodori-TTS が分割生成した音声を1つの WAV に結合
- チャット返答音声を1本の WAV にまとめて UI で扱えるように
- IEEE float(format 3) WAV を手動 RIFF 解析で正しく結合

### 6. 操作性・再生管理の強化
- 返答を選択・ハイライトし、対象の再生／ダウンロード／保存と全体再生速度を制御
- 選択返答の注釈・メタ・音声選択をリロード後も保持
- 選択返答の再生成（regenerate）／削除ボタンを追加
- 過去の返答もフォールバックペイロードで再生成可能に

### 7. キャラクター別のデータ管理
- ログと保存音声をキャラクターごとに管理（起動時に自動マイグレーション）
- Load 時に現在のキャラクターを維持し、コンテキスト上限をキャラ別に設定可能に

### 8. ローカルRAG長期記憶（意味検索）
- 現在の発言に意味的に近い過去の会話を上位数件だけ自動抽出し、【参考：過去の二人の会話の記憶】としてプロンプトへ差し込み。長い履歴でも人格・文脈の統一性を維持
- 埋め込みは `intfloat/multilingual-e5-small` を**完全CPU実行（VRAM不使用）**。fastembed(ONNX) または transformers+torch の2系統を環境に応じて自動選択
- 記憶はキャラクターごとに SQLite 保存。既存のログ・履歴には一切手を加えず、依存が未導入なら自動で無効化（従来動作を維持）
- 「2人だけモード」の記憶は「お題＋話者名」で想起。既存履歴の一括投入スクリプト（`tools/backfill_rag_memory.py`）付き

## 画面モード

### 1Pモード

1Pモードは、1人のキャラクターと会話しながら、LM Studio の応答を Irodori-TTS で読み上げる基本モードです。キャラ設定、TTS Caption、Web検索、話速、感情スタイルを同じ画面で調整できます。

![Rinon Voice Lab 1Pモード](docs/images/rinon-1p-mode.png)

### 2Pキャラモード

2Pキャラモードでは、1Pと2Pのキャラクターを同じ画面に表示し、二人の会話を交互に進められます。2人だけで話すモード、2P音声の別PC生成、キャラクターごとの設定やTTS Captionにも対応しています。

![Rinon Voice Lab 2Pキャラモード](docs/images/rinon-2p-mode.png)

## サポートについて

このリポジトリは個人の実験的な公開物です。サポート、継続メンテナンス、環境ごとの動作保証、個別の導入支援は期待しないでください。参考実装またはローカル実験用として利用してください。

固定ドライブは前提にしていません。`H:` 以外の場所にも配置できます。標準では、このアプリの隣に Irodori-TTS を置きます。

```text
任意のフォルダ\
  RinonVoiceLab\
  Irodori-TTS\
```

## 必要環境

- Windows 10 / 11
- macOS 14 以降の Apple Silicon Mac（実験的対応）
- Python 3.10 以上
- Git
- LM Studio
- LM Studio 側でローカルサーバーを有効化
- ローカル会話モデル
- Irodori-TTS 用の NVIDIA GPU 推奨
- `uv`

標準モデルは `gemma-4-12b-it` を想定しています。31Bモデルも使えますが、VRAM使用量が大きくなります。

macOS では CUDA は使えません。Irodori-TTS は PyTorch の MPS が使える Apple Silicon Mac では `mps`、それ以外では `cpu` で動きます。MPS/CPU では Irodori-TTS の `bf16` は使えないため、`fp32` を使います。音声生成は NVIDIA GPU 環境より遅くなる可能性があります。

Rinon Voice Lab 本体は Python 標準ライブラリだけで動きます。そのため、`requirements.txt` にはアプリ本体用の追加パッケージはありません。Irodori-TTS の依存関係は、Irodori-TTS 専用の仮想環境へインストールします。

## インストールと起動（Windows）

1. このリポジトリをクローン、またはZIPで展開します。
2. LM Studio を起動します。
3. LM Studio の Local Server を有効にします。
4. `gemma-4-12b-it` などの会話モデルを読み込みます。
5. `start_chat_uv.bat` をダブルクリックします。
6. ブラウザで `http://127.0.0.1:7862/` を開きます。

Irodori-TTS が未インストールの場合、`start_chat_uv.bat` が `tools\install_irodori_tts.ps1` を自動実行します。初回は PyTorch やTTSモデルの依存関係が大きいため、時間がかかります。

## インストールと起動（macOS）

1. LM Studio を起動します。
2. LM Studio の Local Server を有効にします。
3. 会話モデルを読み込みます。
4. Terminal でこのリポジトリに移動します。
5. 次を実行します。

```bash
chmod +x start_chat_mac.sh tools/install_irodori_tts.sh
./start_chat_mac.sh
```

6. ブラウザで `http://127.0.0.1:7862/` を開きます。

`start_chat_mac.sh` は、Irodori-TTS が未インストールなら `tools/install_irodori_tts.sh` を実行します。macOS では `uv sync --extra cpu` を使います。この extra は macOS では標準の PyPI PyTorch wheel を使うため、Apple Silicon では MPS が有効な PyTorch であれば `IRODORI_MODEL_DEVICE=auto` によって `mps` が選ばれます。

Python 3.14 では PyTorch wheel が揃わない可能性があるため、macOS スクリプトは標準で Python 3.10 を使います。変更したい場合は次のように指定します。

```bash
IRODORI_PYTHON_VERSION=3.13 ./start_chat_mac.sh
```

Irodori-TTS をアプリの隣以外に置きたい場合:

```bash
IRODORI_ROOT="$PWD/.deps/Irodori-TTS" ./start_chat_mac.sh
```

## Irodori-TTS の手動インストール

アプリフォルダで次を実行します。

```powershell
powershell -ExecutionPolicy Bypass -File tools\install_irodori_tts.ps1
```

標準では CUDA 12.8 用の依存関係を入れます。

```powershell
uv sync --extra cu128
```

CPUだけで試す場合:

```powershell
powershell -ExecutionPolicy Bypass -File tools\install_irodori_tts.ps1 -TorchExtra cpu
```

CPUモードは動作確認用です。音声生成はかなり遅くなる可能性があります。

macOS で手動インストールする場合:

```bash
IRODORI_TORCH_EXTRA=cpu tools/install_irodori_tts.sh
```

## requirements.txt について

`requirements.txt` は、Rinon Voice Lab 本体に直接必要なPythonパッケージがないことを示すためのファイルです。

Irodori-TTS の依存関係は次のどちらかで入れてください。

- `start_chat_uv.bat` から自動インストール
- `tools\install_irodori_tts.ps1` を手動実行

`pip install -r requirements.txt` だけでは Irodori-TTS は入りません。

## 設定

主な環境変数:

| 変数 | 標準値 | 内容 |
| --- | --- | --- |
| `IRODORI_ROOT` | アプリ隣の `..\Irodori-TTS` | Irodori-TTS の場所 |
| `LM_STUDIO_URL` | `http://127.0.0.1:1234/v1` | LM Studio の OpenAI互換API |
| `LM_STUDIO_MODEL` | `gemma-4-12b-it` | 優先モデル名 |
| `LM_STUDIO_CONTEXT_LIMIT` | `8200` | 表示上のコンテキスト上限 |
| `IRODORI_TORCH_EXTRA` | `cu128` | Irodori-TTS インストール時の torch extra |
| `IRODORI_MODEL_DEVICE` | `auto` | Irodori-TTS のモデル実行デバイス。`auto`, `cuda`, `mps`, `cpu`, `xpu` |
| `IRODORI_MODEL_PRECISION` | `auto` | モデル精度。`auto`, `fp32`, `bf16` |
| `IRODORI_CODEC_DEVICE` | `auto` | codec 実行デバイス。通常はモデルと同じ |
| `IRODORI_CODEC_PRECISION` | `auto` | codec 精度。macOS では `fp32` |

## キャラクターデータ

キャラクターは `Character\<character-id>\` の下で管理します。

各キャラクターフォルダには、次のようなファイルやフォルダを置けます。

- `profile.txt`: 手で編集しやすい設定ファイル
- `profile.json`: アプリの保存/読み込み用データ
- `reference\`: 参考音声
- `expressions\<slot>\`: 表情ごとの画像

アプリの `Options` 画面から、キャラ名、キャラ設定、TTS Caption、参考音声、表情画像を編集できます。

## 2PリモートTTS

通常は、1P/2Pの音声を同じPCの Irodori-TTS で生成します。

ツールバーの `TTS PC` で次を選べます。

- `1 PC`: 1P/2Pの音声をこのPCで生成
- `2 PCs`: 1PはこのPC、2Pだけを別PCへ送信

`2 PCs` を選んだ場合は、`2P IP` に2台目のPCを入力します。

例:

- `192.168.0.10`
- `192.168.0.10:7874`
- `http://192.168.0.10:7874`

2台目のWindows PCでは、次のようにリモートTTSサーバーを起動します。

```powershell
$env:IRODORI_ROOT = "H:\AI\Irodori-TTS"
$env:LUVIA_SERVER_PORT = "7874"
python tools\remote_luvia_tts_server.py
```

2台目にも Irodori-TTS がインストールされていて、メインPCからアクセスできる必要があります。

macOS や Linux でリモートTTSサーバーを起動する場合:

```bash
IRODORI_ROOT="$PWD/../Irodori-TTS" \
LUVIA_SERVER_PORT=7874 \
IRODORI_MODEL_DEVICE=auto \
python tools/remote_luvia_tts_server.py
```

## 外部Speakモード

Codex、Claude Code、または別のローカルツールから短いテキストを送り、開いている Rinon Voice Lab のキャラクターUIで読み上げられます。

Rinon Voice Lab を起動して `http://127.0.0.1:7862/` を開いたあと、UTF-8 JSONをPOSTします。

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

主な項目:

| 項目 | 内容 |
| --- | --- |
| `text` | 読み上げるテキスト |
| `emoji` / `emojiStyle` | Irodori の感情絵文字 |
| `caption` / `ttsCaption` | VoiceDesign の演技キャプション |
| `speakerSlot` | `main` または `second` |
| `referencePath` | 任意の参考音声パス |
| `steps` | Irodori の生成ステップ数 |
| `speechRate` | `normal` または `fast` |

ブラウザ側は `/api/speak-events` を監視し、新しいイベントを通常のキャラクターアニメーション、表情切り替え、左右パン、音声保存機能つきで再生します。

## 配布前に消してよい実行時ファイル

次のフォルダはローカル実行時に作られるため、Gitでは無視しています。

- `logs/`
- `profiles/`
- `saved_audio/`
- `static/generated/`
- `__pycache__/`
- `.venv/`

## 動作確認

開発時の簡易チェック:

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

## ライセンス

MIT License です。詳しくは [LICENSE](LICENSE) を参照してください。
