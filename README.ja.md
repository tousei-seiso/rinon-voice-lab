# Rinon Voice Lab 冬星版 日本語README

Rinon Voice Lab は、LM Studio のローカルLLMと Irodori-TTS をつないで、キャラクター会話と音声読み上げを行うローカルアプリです。主な確認環境は Windows ですが、macOS では実験的に起動できるようにしています。

主な機能:

- LM Studio の OpenAI互換ローカルAPIと連携
- Irodori-TTS VoiceDesign による日本語音声生成
- 1P/2P キャラクターの設定、名前、TTS Caption、表情画像を編集
- 会話ログ、セッション、キャラクターデータの保存と読み込み
- 簡易Web検索メモをLLMプロンプトへ追加
- 2P音声だけを別PCの Irodori-TTS へ送るリモートTTSモード

## 本家から改良した主な機能

上の「主な機能」は本家 [sakugetu/rinon-voice-lab](https://github.com/sakugetu/rinon-voice-lab) のものです。
冬星版ではそれに加えて、次の機能を追加・強化しています（各項目の詳細は下の「本家からの主な改良点」を参照）。

- 感情キャプションによるセグメント演技（棒読み回避）
- キャラクター別の CFG Scale / Num Steps / Style Guide による声質の安定化
- 英語→かな正規化と複数かな辞書による日本語TTS品質の向上
- LM Studio 連携の堅牢化（タイムアウト設定・structured output・生成モード切替）
- 分割音声の1本化（1つの WAV に結合）
- 返答の選択再生・再生成・キャラクター別のデータ管理
- 会話履歴の1ターンごと自動保存（会話モード・話者・時刻を保持）
- ローカルCPUで動く RAG 長期記憶（過去会話の意味検索）
- 時系列・網羅・主客に強い3チャネル想起（「一番最初に買ってあげた本は？」「作った料理を全部」に対応）

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
- ターン終わりに VRAM を掃除し、会話前の水準へ巻き戻す（下記）

#### ターン終わりに VRAM を巻き戻す

GPU を掴んでいるのは **2 プロセス**あります。どちらが抱えているかを取り違えると、効かない
対策を足すことになります。そのため掃除のたびに内訳をログへ残します。

**A) 推論サーバ（LM Studio / llama-server, 別プロセス）**
リクエストごとに「スロット」（＝1 本のコンテキスト）を使い、応答後もプロンプトの KV
キャッシュを載せたまま抱えます。さらに**同時リクエストが来ると並列用のスロットを追加確保し、
空いても手放しません**。返答後に走る RAG 処理（検索クエリの書き換え・事実台帳の抽出）は
バックグラウンドスレッドから飛ぶため、次の発言の本文生成と重なりやすいです。
ただし llama.cpp は KV バッファをモデルロード時に `n_ctx` ぶん確保しきるため、
**`erase` で戻るのは「どこまでキャッシュ済みか」の帳簿だけで、VRAM のバイト数は減りません**
（実測: `erase` 成功後も 12.6GB のまま）。ここで効くのは「そもそも 2 本目のスロットを
確保させない」直列化の方です。

**B) Irodori-TTS（このプロセス内, `IRODORI_MODEL_DEVICE=auto` なら cuda）**
torch のキャッシュアロケータは解放済みブロックをドライバへ返さず抱え続けます。文の長さごとに
違うサイズのブロックが溜まるため、返答を合成するほど増えて戻りません。`empty_cache()` で
明示的に返させます。**こちらは実際に VRAM が減ります。**

対策は 5 段構えで、いずれも「非対応なら黙って諦める」層に留めています。

1. **直列化**（`LM_SERIALIZE_REQUESTS=1`）… 補助生成が本文生成へ割り込まないようにし、
   そもそも 2 本目のスロットを確保させない。サーバ側の居残りに効くのは実質これだけ。
2. **キャッシュ無効化**（`LM_CACHE_PROMPT=aux`）… 補助生成のリクエストへ
   `cache_prompt=false` を付け、プロンプト KV を残させない。補助生成のプロンプトは毎回
   中身が違って再利用が効かないうえ、載ると返答本文が使いたい履歴の接頭辞を追い出します。
   サーバが未知フィールドを 400 で弾いたら自動で外して再送し、以後は付けません。
3. **サーバ側の明示解放**（`LM_KV_CACHE_RELEASE=idle`）… LM リクエストと TTS 合成が
   途切れて `LM_KV_CACHE_RELEASE_DELAY` 秒静まったら、`POST /slots/{id}?action=erase` で
   全スロットの KV キャッシュを捨てさせる。llama-server は `/slots` を有効にしておく必要が
   あります（この API を持たないサーバでは 1 度試して以後黙ります）。上記のとおり
   バイト数の巻き戻りは期待できません。
4. **自プロセスの解放**（`VRAM_RELEASE_TORCH=1`）… 同じ掃除のタイミングで `gc.collect()` →
   `torch.cuda.empty_cache()` を呼び、TTS が抱えた CUDA キャッシュをドライバへ返します。
   合成の真っ最中は `Irodori_lock` が空いていないので触りません（これから使い回すブロックを
   奪わないため）。TTS をリモートへ出す構成や CPU 実行では自動的に no-op になります。
5. **置き場所の切り替え**（`VRAM_TTS_MIN_FREE_MIB=3000`）… 1〜4 はどれも「溜まったぶんを
   返す」対策なので、推論サーバがモデルロード時に確保しきったぶんには効きません。そこで
   合成の直前に空き VRAM を測り、この値を割っていたらそのターンの TTS を **CPU へ逃がします**
   （精度も `bf16` 固定を捨ててデバイスなりに解決し直します）。合成は遅くなりますが、
   溢れさせて GPU ごと詰まらせる（後述）よりは確実にましです。`0` で無効。適正値は
   下記ログの `peak=` に余白を足して決めてください。

##### 切り分け方（ログの読み方）

`VRAM_MEMORY_LOG=1`（既定）で、掃除の前後に次の 1 行が出ます。

```
[vram] before release (idle): total=11868/12282MiB  self-torch reserved=2048MiB allocated=400MiB peak=2048MiB
[vram] torch cache released: 1536MiB (reserved 2048 -> 512MiB)
[lm] KV cache released: 1 slot(s) (idle)
[vram] after  release (idle): total=10310/12282MiB  self-torch reserved=512MiB allocated=400MiB peak=2048MiB
```

- `self-torch reserved` が減って `total` も減った → 犯人は **TTS（このプロセス）**。これで解決。
- `self-torch reserved` が小さいのに `total` が大きい → 犯人は **推論サーバ側**。`erase` では
  戻らないので、サーバの起動設定（並列リクエスト数・文脈長・KV キャッシュ量子化など）で
  削るか、モデルをアンロードするしかありません。
- `peak` は起動からの `reserved` 最大値で、`empty_cache()` で返したあとも残ります。
  これが「TTS が GPU で要る量」なので、`VRAM_TTS_MIN_FREE_MIB` はこの値に余白を足して
  決めます。安全弁が働いたターンには次の 1 行が出ます。

```
[vram] low headroom before request: free=2680MiB < 3000MiB (TTS would fall back to cpu)
[vram] TTS falls back to cpu: free=2680MiB < 3000MiB -- ...
[vram] TTS back on cuda: free=4620MiB >= 3000MiB

```

Windows(WDDM) の nvidia-smi はプロセス別の使用量を報告しない（GeForce では
`--query-compute-apps` が空で返る）ため、自プロセスぶんは torch から直接読んでいます。

##### 軽い量子化へ替えたら悪化する場合

モデルを小さくすると llama.cpp は空いた VRAM を全層オフロード・並列スロット・プロンプト
キャッシュで使い切るため、**モデルが軽くなってもサーバの占有はむしろ増えます**。
実測（RTX 4070 / 12282MiB, `--ctx-size 8192 --parallel 2 --batch-size 2048`）:

| モデル | ファイル | サーバ占有 | プロンプト評価 | 生成 |
| --- | --- | --- | --- | --- |
| `gemma-4-12b-it-Q6_K` | 9.11GB | 一部の層は CPU に残る | 227 tok/s | 14 tok/s |
| `gemma-4-12B-it-QAT-Q4_0` | 6.50GB | 9436〜11498MiB（全層 GPU） | 1398 tok/s | 19 tok/s |

QAT 版は速い代わりに VRAM の余白が消えるので、そこへ TTS を載せると溢れます。溢れたぶんは
NVIDIA ドライバがシステムメモリへ退避させるため、**確保自体は成功するのに** GPU 演算が
PCIe 律速になり、プロンプト評価が **1398 → 5.4 tok/s**（4159 トークンの評価に 765 秒）まで
落ちました。同じ GPU でデスクトップを描いているので、画面更新もマウスカーソルも止まります。
アプリ側はどれだけ待っても返ってこないため `LM_STUDIO_TIMEOUT` まで粘って落ちます。

サーバ側（LM Studio なら Load 設定）で削る順:

1. **Max Concurrent Predictions を 1 に**… `--parallel 2` はスロットを 2 本確保します。
   アプリは直列化していますが、llama.cpp は直列でも LRU でスロットを交互に選ぶため
   両方に KV が居残ります（サーバログの
   `making room for prompt cache entry, removing oldest entry (size = 1693.788 MiB)`）。
2. **GPU オフロード層数を下げる**… 1〜2GB 単位で空きを作れます。
3. **評価バッチサイズを 2048 → 512**… prefill の計算バッファが縮みます。
4. **Vision(mmproj) を切る**… テキストだけ使うなら `--mmproj` のぶんは丸ごと無駄です。

あわせて NVIDIA コントロールパネルの「CUDA - システムメモリ フォールバック ポリシー」を
**「フォールバックなしを優先」** にしておくと、溢れたときにフリーズではなく CUDA OOM で
即座に失敗します（デスクトップが固まらず、原因もその場で分かります）。

##### 速度とのトレードオフ

サーバ側を解放したターンの次の返答は、プロンプトを再評価するぶんだけ待ちが増えます。
速度優先なら `LM_KV_CACHE_RELEASE_DELAY` を伸ばす（連投中はキャッシュを温存）か
`LM_KV_CACHE_RELEASE=off` にしてください。`VRAM_RELEASE_TORCH` 側は次の合成で確保し直す
ぶんだけ（数十ミリ秒程度）増えます。

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
- 想起の的中率を高める作り込み:
  - 会話モード（1P通常／2人だけモード）と話者スロットで想起を絞り込み、2Pのお題が1P会話へ漏れ出すのを防止
  - ユーザー発話をそのまま検索せず、LLM で焦点を絞った検索クエリへ書き換え（挨拶・依頼の枕詞を除去、「〜以外」などの否定語を排除、行為の主体・客体を保持し、主語省略時のみユーザーを主体に補完）
  - 「全部挙げて」等の列挙・網羅質問では観点違いの複数クエリを生成して和集合で取得し、ユーザーが明示的に求めたときだけ記憶を漏れなく列挙
  - 近重複の集約、`top_k` / `min_score` の調整、圧縮後コンテキスト基準の重複除去により、長い履歴で文脈から溢れた過去も取りこぼさない
- 再構築・診断ツール: `tools/rebuild_from_chatlog.py`（`logs/chat.jsonl` を正本に履歴と記憶DBを元の時刻付きで再生成。感情キャプションも各ターンの `segments` から組み直す）、`tools/diagnose_recall.py`（想起スコアを一覧して当たり外れを診断）

### 9. 3チャネル想起（時系列・網羅・主客）
意味検索（cosine top-k）だけでは構造的に答えられない質問があります。「一番最初に買ってあげた本は？」は時間を見ないので最古の1件を保証できず、「作った料理を全部」は該当が上位k件を超えた時点で必ず溢れます。そこで用途の違う3つのチャネルを併用します。

- **ベクトル（意味）**: 従来の類似度検索。話題の近い記憶を拾う
- **語彙（全件一致・上限なし）**: SQLite の FTS5(trigram) 索引と `LIKE` のハイブリッド。順位ではなく一致で拾うので件数の窓による漏れが出ない
  - trigram は3文字未満に反応しないため、3文字以上は FTS5、1〜2文字（「本」等の漢字1字）は `LIKE` に振り分け
  - 活用差を吸収する語幹化（「買った」→「買」で「買ってあげた」にも一致）。日本語のクエリと本文の活用ズレによる取りこぼしを防止
- **事実台帳（集計）**: 往復から「誰が・誰に・何を・どうした」を抽出し `facts` テーブルへ正規化。列挙は検索ではなく `SELECT DISTINCT` なので**全件が返る**
  - 主体・客体・**行為の向き**（`user->char` / `char->user`）を構造として保持するため、「俺が君に作ってあげた料理」と「君が俺に作ってくれた料理」を取り違えない
  - 抽出はハイブリッド。日本語の授受表現（「〜してあげた」＝発話者→相手、「〜してくれた」＝相手→発話者）と `role` の組み合わせで大半をLLMなしに確定し、決まらない往復だけローカルLLMへ回す
  - 判定できなかった要素は捨てず「主客不明」として保持（捨てると「台帳に無い＝存在しない」という別の漏れになる）
  - 台帳は索引であって正本ではないので、プロンプトには必ず**出典の原文**も併記し、最終判断の根拠を原文に置く
  - 記録するのは「後から何があったかを思い出せる事実」だけ。実ログでの検証をもとに次を除外:
    - **客体の無い事実**（`言う: （空）`）— 何も思い出せず、列挙の枠を食い潰すだけ
    - **「言う」「見る」**（発話・知覚）— 会話ログ自体が「言ったこと」の記録なので冗長。実測ではゴミ客体の最大の発生源だった（約束・告白のように後から参照する内容は LLM 側が拾う）
    - **中身の無い客体** — 助詞・形式名詞・代名詞（`の` `こと` `気` `私`）や `挨拶` `気持ち` `顔` `言葉` など
    - **第三者が絡む行為の向き** — 受け手がユーザーでもそのキャラでもない場合は `unknown`（主客不明）にする。事実は残すが「俺が君にした事」の絞り込みには混ぜない
    - **名前として通らない主体** — 日本語の「が」は主格だけでなく動詞の一部にもなるため（「立ち上がって」）、素朴に「が」の直前を取ると節の断片が主体になる（実測: 主体が `翌朝になって何とか立ち上`）。表記が混ざる候補・時間表現・動詞を含む候補は主体にせず「主体不明」で残す
    - **節の断片を客体にしない** — 連用形・連体修飾・時間表現は繰り返し剥がす（実測: `買う: 行って薬`→`薬` / `行く: 倒れて不調なままネパールの病院`→`ネパールの病院` / `登る: 翌朝` は消える）。1字の動詞語幹（`行く: 見`）も落とす
    - **離れた授受表現** — 間に別の格助詞がある授受は別の述語のもの（実測: 「病院に行って薬を**もらった**」で、行く行為が「ルリが行った」になった）
    - **目的の「に」** — 「〜に行く」の「に」は場所だけでなく目的も表すため、場所らしくない客体は落とす（実測: `行く` の30件中12件が `気分転換` `任務` `分析` `買い物` のような目的だった）。方向しか表さない「へ」は検査しない。「友達に会った」の「に」は相手なので `会う` は対象外
    - **LLM が同じ行為を取れている往復のルール由来の弱い事実** — 客体が包含関係にある重複（LLMの`麻辣麻婆豆腐`とルールの`からさの花椒をたっぷりかけた麻辣麻婆豆腐`）は LLM 側を採る
  - 実測（771往復）では、日常会話に授受表現が少ないため**約87%の往復が LLM 抽出を要する**。`--rule-only` だけでは台帳がほぼ埋まらないので、構築時は LM Studio を起動して LLM 抽出ありで実行する
  - **相（modality）と出来事の時刻**も向きと同じく構造で持つ。日本語の動詞は語幹が同じまま時制・法だけ変わるため、表層パターンだけでは次を取り違える（いずれも実測）:
    - 未来の予定「次はうどんなカルボナーラを作ってあげるね」→ *作った料理*として列挙された
    - 否定「カルボナーラを作れなかった」→ *作った料理*として残った
    - 回想「1年前に多摩川の花火大会に行ったよね」→ *その話をした日*にした事になった
  - `modality`（`done` / `plan`予定 / `wish`願望・仮定 / `negated`しなかった / `unknown`）は**その動詞の直後の語尾だけ**を見て決める（文全体を見ると別の述語の時制を持ち込む。「作ってあげるって約束した」の「した」で done にしない）。非過去の授受表現（「作ってあげる」「買ってきてくれる？」）は申し出・依頼であって完了ではない
  - `occurred` は出来事そのものの時期。「1年前」「去年の夏」「昨日」「明日」「来月」を往復の `ts` を基準日にして解決し、粒度つきの前方一致文字列（`2025` / `2025-07` / `2025-07-24`）で持つ。原文の言い方は `time_hint` に残して提示にも出す（`ts` は**その話をした日**にすぎない）
  - 年の書かれていない表現（「8月5日」「秋」）は**相によって向きが逆**になる。回想なら直近の過去、予定なら次に来るその日。ここを分けないと来週の予定が去年の出来事になる（実測: 2026-07-29 の「8月5日に抜糸」が `2025-08-05` と記録された）。`time_hint` が残っているので、規則を直したら `--fix-occurred` でLLM無しに再計算できる
  - マーカーの無い非過去は `done` と断定せず `unknown` にし、想起では done と一緒に返して提示で「（実行したか不明）」と明示する。**予定を done と誤るのと同じくらい、done を予定と誤って列挙から落とすのが危険**なため
  - 想起の既定は `done`（実際にした事）。予定は捨てずに一覧の別節「＜まだしていない事＞」へ出すので、「今度作ってくれるって言ってた料理」にも答えられる。期間指定は `occurred` と `ts` のどちらかが窓に入れば該当（片方だけで見ると回想が期間検索から丸ごと漏れる）
  - 「俺がした事」を訊かれたら、**その人自身の行為**（`direction='self'`。「俺が薬局に行った」）も併せて引く。self は相手への行為ではないので向きの一致では拾えず、これが無いと自分の行動が丸ごと漏れる（実測: 台帳474件のうち165件が self で、全部落ちていた）
  - 「した事」を訊かれたときも、同じ行為の予定・願望は**打ち消し材料として**別節に併記する（`RAG_LEDGER_PLANS`）。一覧から黙って落とすと、年表に残る原文（「次は〜作ってあげるね」）だけを読んだLLMが「作った料理」として挙げてしまうため
  - 既存の台帳には列を後付けするだけで動く（既存行は `modality='done'` / `occurred=''` ＝従来と同じ挙動）。ただし**すでに done として入っている予定はそのまま残る**ので、直すには再抽出が必要（下記）

**時系列の想起**（タイムスタンプの活用）
- 「一番最初／初めて」「最後／最近」「いつ」「去年の夏」「3月」「2年前」などを正規表現で検出し（クエリ書き換えLLMの意図分類でも補完）、スコア上位プールを確保してから時刻順に並べ替えて選抜
- プロンプトへは**古い順の年表**として渡す。各行に日付・経過期間（「約1年7ヶ月前」＝サーバ側で計算済み。LLMに日数計算をさせない）を添え、先頭が最古・末尾が最新であることを明示
- 「記録の範囲」も併記し、範囲より前は「記録が無い」だけで「出来事が無かった」ことにはならないと伝える（記録上の最初を本当の最初と断定させない）
- 期間表現は `since`/`until` へ解決して検索側で絞り込み

**ツール**
- `tools/build_fact_ledger.py`: 既存ログから台帳を一括構築（`--rule-only` でLLM不使用、`--dry-run` で確認、中断しても未抽出分から再開）。確認・修復もここから: `--list --modality plan`（予定だけ）／`--list --object カルボナーラ`（あの品が入っているか）／`--fix-occurred`（出来事の時期だけを `time_hint` から再計算。LLM不要・数秒）
- `tools/check_fact_modality.py`: 相（した事／予定／しなかった事）と出来事時刻の判定を例文で回帰チェック（LLM も DB も使わないので数百ミリ秒。`fact_extract.py` の語尾・時間表現の規則を触ったら実行する）
- `tools/sync_memory.py`: 履歴を編集したあとに記憶系ファイルを**差分同期**（下記）
- `tools/repair_chatlog_segments.py`: `chat.jsonl` の `segments` に欠けた分割本文を `chunks`/`audios` から復元（感情セグメント導入当日のごく一部のターンが対象。裏付けが取れたターンだけ書き込む）
- `tools/diagnose_temporal.py`: 時系列・列挙の想起を LLM 抜きで検証し、意図検出／ベクトル／語彙／台帳のどこで落ちているかを切り分け（ユーザー名・キャラ名はアプリ本体と同じくプロファイルから補完する。ここが空だと自分の行為＝`self` を拾えず本番と違う結果になり、診断が誤誘導する）
- `tools/audit_memory.py`: `chat.jsonl`・`history.json`・`memories` の件数を突き合わせ、**保存漏れ**（何をしても答えられない漏れ）と ts の健全性、孤児事実、台帳の抽出率を監査

### 履歴を編集したあとの同期
履歴系ファイルには上下関係があります。

```
logs/chat.jsonl                                ← 一番大本の正本（全ターンの生ログ）
  ├→ profiles/sessions/<charId>/history.json     画面復元用のキャラ別履歴
  ├→ profiles/sessions/<charId>/memory.sqlite3   RAG検索DB＋事実台帳
  └→ logs/chat_emotion.jsonl                     感情キャプション付き返答
```

`tools/sync_memory.py` が、編集した場所に応じて下流を**差分だけ**揃えます（全往復の埋め込みを計算し直さないので高速で、事実台帳も保たれます）。

| 編集した場所 | コマンド | 揃える対象 |
|---|---|---|
| **UIから会話を削除** | **不要（削除時に自動で全系統へ反映）** | 4系統すべて |
| `chat.jsonl` を手修正 | `sync_memory.py --sync-emotion --extract` | history.json / memory.sqlite3 / chat_emotion.jsonl / 台帳 |
| `history.json` を手修正 | `sync_memory.py --source history --propagate --extract` | memory.sqlite3 / 台帳 ＋ **大本の chat.jsonl と chat_emotion.jsonl へ削除を逆伝播** |
| 全部作り直す（chat.jsonl が正本） | `rebuild_from_chatlog.py --reset --sync-emotion` → `build_fact_ledger.py` | 全部 |
| 全部作り直す（履歴が正本） | `rebuild_rag_from_history.py --reset --extract` | memory.sqlite3 / 台帳 |

```bash
python tools/sync_memory.py --dry-run                                   # まず差分の確認
python tools/sync_memory.py --sync-emotion --extract --rule-only        # chat.jsonl を正本に同期

python tools/sync_memory.py --source history --propagate --dry-run      # UI削除後の確認
python tools/sync_memory.py --source history --propagate --extract --rule-only
```

> **相（予定・回想）の列を入れたあとの作り直し**: 列は自動で後付けされますが、既存の事実は `modality='done'` / `occurred=''` のままなので、**すでに「した事」として入っている予定や回想はそのままです**。直すには再抽出してください。
> ```bash
> python tools/check_fact_modality.py                     # まず判定が期待どおりか確認（LLM不要・数百ミリ秒）
> python tools/build_fact_ledger.py --redo-rule --dry-run  # ルール由来の事実を作り直すと何が入るか（LLM不要）
> python tools/build_fact_ledger.py --redo-rule            # ルール由来を作り直す（数秒。LLM 由来は残る）
> python tools/build_fact_ledger.py --list --modality plan  # 予定として記録された分を目視確認
> ```
> LLM 由来の事実にも相を付けたい場合は、その分だけ作り直します（LM Studio が必要）。全期間だと数十分かかるので、期間や動詞で絞るのが実用的です。
> ```bash
> python tools/build_fact_ledger.py --redo-verb 作る        # 「作る」に関わる往復だけをやり直す
> python tools/build_fact_ledger.py --redo --since 2026-07-01
> ```

差分同期がすること: 正本に無い往復をDBから削除（**その往復から抽出した事実も一緒に削除**）／DBに無い往復を追加（埋め込み計算は差分だけ）／時刻の変更を反映（台帳の日付も揃える）／出典を失った事実（孤児）を掃除。照合は「ユーザー発言＋返答＋会話モード」の一致で行います（`ts` は手修正されうるので照合キーに含めません）。

> **感情キャプションも `chat.jsonl` だけから復元されます**: `chat.jsonl` の各ターンは `segments`（`{style, emoji, text}` の並び）を持つため、表示用の注釈文「（😊嬉しそうに…）本文」は `emotion_caption.build_annotated_reply` で組み直せます。`chat_emotion.jsonl` の `annotatedReply` はこの関数の出力そのもので、キャプションの独立した情報源ではありません（実測で全件一致）。そのため `--sync-emotion` は「残る返答だけへ絞る」のではなく **`chat.jsonl` から作り直します**（消えた行を落とすだけでなく、欠けている行を足し、注釈文を組み直す）。例外は感情セグメント導入当日のごく一部のターンで `segments` に `text` が無く、これは `tools/repair_chatlog_segments.py` で補修できます。

> **UIの削除について**: 削除は往復単位（あなたの発言＋返答）で行われ、その場で **4系統すべて**（`history.json` / `chat.jsonl` / `chat_emotion.jsonl` / `memory.sqlite3` と事実台帳）から取り除かれます。同期スクリプトを流す必要はありません。生ログの書き換え前には `.bak` へ退避します。2人だけモードで同じお題に他の返答が残る場合は、お題を残して返答だけを消します。生成済みの音声ファイル（`static/generated`）は他から参照される可能性があるため消しません。
>
> この削除は自動保存（`/api/session`）とは**別経路**（`/api/delete-turn`）で行います。自動保存は「いまの会話コンテキスト」を書くだけなので、そこから削除を推論すると Clear Context（コンテキストのリセット）と区別できず、残すべき生ログまで消してしまいます。`history.json` と `chat.jsonl` は内容が一致しないのが正常です（前者は Clear Context で空になり、後者は残る）。
>
> **`--propagate` が必要な理由**: UIの削除は `history.json` にしか効かないので、一番大本の `chat.jsonl` には残ります。そのままだと後日 `rebuild_from_chatlog.py` を実行したときに削除した会話が復活します。`--propagate` は大本まで消して整合させます（実行前に `.sync.bak` へ退避）。なお `--propagate` は `chat.jsonl` 全体を見るため、`--char` で対象を絞らずに実行してください。
>
> **`--reset` を伴う全再構築の注意**: DB ファイルごと削除するため事実台帳も消えます（FTS5索引は自動で作り直されます）。`rebuild_rag_from_history.py` は `--extract` で台帳まで作り直せます。DBを作り直すと `memories.id` が振り直されるため、古い `source_id` を持つ事実は自動で掃除されます。

### 10. 会話履歴の自動保存と話者識別
- 会話が1ターン進むごとにセッション履歴を自動保存（明示保存を待たずに復元できる）
- 履歴に会話モード・話者・タイムスタンプを保持し、オートセーブでもモードを取り違えない
- 話者をスロット（1P=main／2P=second）で識別し、1Pと2Pで同名のキャラクターでも記録・想起が混ざらない

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

標準モデルは `gemma-4-12b-it` と Google 公式 QAT の `google/gemma-4-12b-qat` を想定しています（`LM_STUDIO_MODEL` で切り替え）。31Bモデルも使えますが、VRAM使用量が大きくなります。

macOS では CUDA は使えません。Irodori-TTS は PyTorch の MPS が使える Apple Silicon Mac では `mps`、それ以外では `cpu` で動きます。MPS/CPU では Irodori-TTS の `bf16` は使えないため、`fp32` を使います。音声生成は NVIDIA GPU 環境より遅くなる可能性があります。

Rinon Voice Lab 本体は Python 標準ライブラリだけで動きます。そのため、`requirements.txt` にはアプリ本体用の追加パッケージはありません。Irodori-TTS の依存関係は、Irodori-TTS 専用の仮想環境へインストールします。

## インストールと起動（Windows）

1. このリポジトリをクローン、またはZIPで展開します。
2. LM Studio を起動します。
3. LM Studio の Local Server を有効にします。
4. `gemma-4-12b-it` または `google/gemma-4-12b-qat` などの会話モデルを読み込みます。
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

| 変数 | 標準値 | 内容 | 本家 | 冬星版 |
| --- | --- | --- | :-: | :-: |
| `IRODORI_ROOT` | アプリ隣の `..\Irodori-TTS` | Irodori-TTS の場所 | ✅ | ✅ |
| `LM_STUDIO_URL` | `http://127.0.0.1:1234/v1` | LM Studio の OpenAI互換API | ✅ | ✅ |
| `LM_STUDIO_MODEL` | `gemma-4-12b-it@q6_k` | 優先モデル名。対応系列（`gemma-4-12b-it` / `google/gemma-4-12b-qat`）以外を指定すると使われず、`LM_STUDIO_DEFAULT_MODEL` → `gemma-4-12b-it@q6_k` の順に落ちる。GGUF のフルパス指定も可 | ✅ | ✅ |
| `LM_STUDIO_DEFAULT_MODEL` | （なし） | `LM_STUDIO_MODEL` が対応系列外・未設定のときに使うモデル名 | — | ✅ |
| `LM_STUDIO_ALLOWED_MODELS` | （なし） | 対応系列に加えて一時的に許可するモデル名（カンマ区切り・完全一致）。ただし思考タグは既知の書式でしか落とせないため、恒久的に増やすときは `app.py` の `LM_MODEL_CATALOG` へ、そのモデルの思考タグの書式（`LM_THINKING_FORMATS`）と対で追記する | — | ✅ |
| `LM_STUDIO_CONTEXT_LIMIT` | `8200` | 表示上のコンテキスト上限 | ✅ | ✅ |
| `LM_COMPACT_CONTEXT_LIMIT` | `4200` | プロンプトへ載せる会話履歴の実上限（文字数）。画面の context 上限はこの値でクランプされる | ✅ | ✅ |
| `LM_RECENT_MESSAGE_COUNT` | `12` | 要約に畳まず原文のまま載せる直近の発言数 | ✅ | ✅ |
| `LM_CONTEXT_LENGTH` | `0` | モデルの文脈長（トークン）を手動指定。`0` なら LM Studio の `/api/v0/models` から自動取得 | — | ✅ |
| `LM_OUTPUT_RESERVE_TOKENS` | `1280` | 思考＋本文のために必ず空けておくトークン数。空返答が出るなら増やす | — | ✅ |
| `LM_CONTEXT_PROBE_TTL` | `60` | 自動取得した文脈長のキャッシュ秒数 | — | ✅ |
| `LM_MEMORY_DIGEST` | `1` | 記憶ブロックが枠を超えたとき、別呼び出しで要点メモへ圧縮する（`0` なら行単位の間引きだけ） | — | ✅ |
| `LM_MEMORY_DIGEST_MAXTOK` | `700` | 要点メモ 1 回ぶんの生成上限 | — | ✅ |
| `LM_MEMORY_DIGEST_CHUNKS` | `4` | 記憶が文脈に収まらないときの分割要約の最大チャンク数 | — | ✅ |
| `LM_SERIALIZE_REQUESTS` | `1` | 推論サーバへのリクエストを 1 本ずつに直列化（`0` で従来どおり並行）。VRAM 対策 | — | ✅ |
| `LM_CACHE_PROMPT` | `aux` | `cache_prompt=false` を付ける範囲。`aux`＝補助生成だけ / `all`＝返答本文も / `off`＝付けない | — | ✅ |
| `LM_KV_CACHE_RELEASE` | `idle` | KV キャッシュの明示解放。`idle`＝リクエストが途切れたら / `each`＝毎回 / `off`＝しない | — | ✅ |
| `LM_KV_CACHE_RELEASE_DELAY` | `2` | `idle` で解放するまでの待ち秒数。伸ばすと連投中はキャッシュを温存する | — | ✅ |
| `LM_KV_CACHE_RELEASE_TIMEOUT` | `5` | 解放 API の待ち時間（秒） | — | ✅ |
| `VRAM_RELEASE_TORCH` | `1` | 掃除時に自プロセス（Irodori-TTS）の torch CUDA キャッシュも `empty_cache()` で返す | — | ✅ |
| `VRAM_MEMORY_LOG` | `1` | 掃除の前後に GPU 使用量の内訳（全体＋自プロセスの torch 保持量）をログへ出す | — | ✅ |
| `VRAM_TTS_MIN_FREE_MIB` | `3000` | 空き VRAM がこれを割ったら TTS をそのターンだけ CPU へ逃がす（`0` で無効）。適正値は `[vram]` ログの `peak=` ＋余白 | — | ✅ |
| `VRAM_FREE_PROBE_TTL` | `3` | 空き VRAM を測り直す間隔（秒）。合成チャンクごとに `nvidia-smi` を起こさないための間隔 | — | ✅ |
| `IRODORI_TORCH_EXTRA` | `cu128` | Irodori-TTS インストール時の torch extra | ✅ | ✅ |
| `IRODORI_MODEL_DEVICE` | `auto` | Irodori-TTS のモデル実行デバイス。`auto`, `cuda`, `mps`, `cpu`, `xpu` | ✅ | ✅ |
| `IRODORI_MODEL_PRECISION` | `auto` | モデル精度。`auto`, `fp32`, `bf16` | ✅ | ✅ |
| `IRODORI_CODEC_DEVICE` | `auto` | codec 実行デバイス。通常はモデルと同じ | ✅ | ✅ |
| `IRODORI_CODEC_PRECISION` | `auto` | codec 精度。macOS では `fp32` | ✅ | ✅ |
| `RAG_MEMORY_ENABLED` | `1` | RAG長期記憶の有効化（`0`で無効・従来動作） | — | ✅ |
| `RAG_EMBED_MODEL` | `intfloat/multilingual-e5-small` | 埋め込みモデル | — | ✅ |
| `RAG_RECALL_TOP_K` | `16` | 1発言あたり想起する記憶の最大件数 | — | ✅ |
| `RAG_RECALL_MIN_SCORE` | `0.75` | 想起の類似度しきい値 | — | ✅ |
| `RAG_RECALL_DEDUP` | `1` | 近重複記憶の集約（`0`で無効） | — | ✅ |
| `RAG_QUERY_REWRITE` | `1` | 検索クエリのLLM書き換え（`0`で無効・原文検索） | — | ✅ |
| `RAG_QUERY_REWRITE_MODE` | （空） | 書き換えLLMの生成モード（空でチャットに追従。重ければ `prefill`） | — | ✅ |
| `RAG_QUERY_REWRITE_MULTI` | `3` | 列挙質問で生成する検索クエリの最大本数（`1`で単一） | — | ✅ |
| `RAG_LEXICAL_ENABLED` | `1` | 語彙チャネル（FTS5+LIKE）の有効化（`0`でベクトルのみ） | — | ✅ |
| `RAG_LEXICAL_LIMIT` | `24` | 語彙チャネルから載せる最大件数 | — | ✅ |
| `RAG_LEXICAL_LIMIT_ENUM` | `48` | 列挙質問時の語彙チャネル上限 | — | ✅ |
| `RAG_LEXICAL_SLACK` | `0.03` | 語彙一致行に許す類似度の緩め幅（独立した証拠なので閾値を下げる） | — | ✅ |
| `RAG_TEMPORAL_POOL_K` | `64` | 時系列で並べ替える前に確保する候補プール（狭いと最古/最新を取りこぼす） | — | ✅ |
| `RAG_TEMPORAL_K` | `8` | 年表としてプロンプトへ載せる件数 | — | ✅ |
| `RAG_TEMPORAL_BAND` | `0.02` | 時系列選抜で「話題の芯」と見なすスコア帯（実測: e5 のスコアは 0.80〜0.84 に潰れるため広げると無関係な古い記憶が「最初」を奪う） | — | ✅ |
| `RAG_LEDGER_ENABLED` | `1` | 事実台帳の有効化（`0`で読み書きしない） | — | ✅ |
| `RAG_LEDGER_LIMIT` | `60` | 台帳から載せる事実の上限 | — | ✅ |
| `RAG_LEDGER_TURNS` | `8` | 台帳の裏付けとして年表へ含める原文の件数 | — | ✅ |
| `RAG_LEDGER_PLANS` | `12` | 「した事」を訊かれたときに打ち消し材料として併記する予定・願望の件数（`0` で併記しない） | — | ✅ |
| `RAG_LEDGER_ALWAYS` | `0` | 時系列・列挙以外でも常に台帳を併用する | — | ✅ |
| `RAG_LEDGER_LIVE` | `1` | 返答後に今回の往復を台帳へ増分抽出（別スレッド） | — | ✅ |
| `RAG_LEDGER_LIVE_LLM` | `1` | 増分抽出でLLMを使う（`0`ならルール抽出のみ・完全に無料） | — | ✅ |
| `RAG_FACT_EXTRACT_MAXTOK` | `256` | 事実抽出LLMの生成上限 | — | ✅ |

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
