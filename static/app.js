const messagesEl = document.querySelector("#messages");
const composer = document.querySelector("#composer");
const messageInput = document.querySelector("#message");
const sendButton = document.querySelector("#send");
const sendShortcut = document.querySelector("#sendShortcut");
const autoStartButton = document.querySelector("#autoStart");
const player = document.querySelector("#player");
const regenAudioButton = document.querySelector("#regenAudio");
const saveAudioButton = document.querySelector("#saveAudio");
const deleteMessageButton = document.querySelector("#deleteMessage");
const audioSaveStatus = document.querySelector("#audioSaveStatus");
const portraitWrap = document.querySelector("#mainPortraitWrap");
const secondPortraitWrap = document.querySelector("#secondPortraitWrap");
const portrait = document.querySelector("#portrait");
const secondPortrait = document.querySelector("#secondPortrait");
const mainCharacterNameLabel = document.querySelector("#mainCharacterNameLabel");
const secondCharacterNameLabel = document.querySelector("#secondCharacterNameLabel");
const modelSelect = document.querySelector("#model");
const stepsInput = document.querySelector("#steps");
const speechRate = document.querySelector("#speechRate");
const speechRateButtons = Array.from(document.querySelectorAll("[data-rate]"));
const replyLength = document.querySelector("#replyLength");
const llmGenerationMode = document.querySelector("#llmGenerationMode");
const systemPrompt = document.querySelector("#systemPrompt");
const userAddress = document.querySelector("#userAddress");
const ttsCaption = document.querySelector("#ttsCaption");
const secondSystemPrompt = document.querySelector("#secondSystemPrompt");
const secondTtsCaption = document.querySelector("#secondTtsCaption");
const contextLimit = document.querySelector("#contextLimit");
const contextUsage = document.querySelector("#contextUsage");
const autoEmoji = document.querySelector("#autoEmoji");
const webSearch = document.querySelector("#webSearch");
const twoPlayerMode = document.querySelector("#twoPlayerMode");
const twoOnlyMode = document.querySelector("#twoOnlyMode");
const ttsBackendMode = document.querySelector("#ttsBackendMode");
const secondTtsHost = document.querySelector("#secondTtsHost");
const mainCharacterSelect = document.querySelector("#mainCharacterSelect");
const secondCharacterSelect = document.querySelector("#secondCharacterSelect");
const openOptionsButton = document.querySelector("#openOptions");
const closeOptionsButton = document.querySelector("#closeOptions");
const shutdownAppButton = document.querySelector("#shutdownApp");
const optionsModal = document.querySelector("#optionsModal");
const editCharacterSelect = document.querySelector("#editCharacterSelect");
const newCharacterButton = document.querySelector("#newCharacter");
const loadCharactersButton = document.querySelector("#loadCharacters");
const saveCharactersButton = document.querySelector("#saveCharacters");
const editCharacterName = document.querySelector("#editCharacterName");
const editSystemPrompt = document.querySelector("#editSystemPrompt");
const editTtsCaption = document.querySelector("#editTtsCaption");
const editStyleGuide = document.querySelector("#editStyleGuide");
const editSteps = document.querySelector("#editSteps");
const editCfgScaleText = document.querySelector("#editCfgScaleText");
const editCfgScaleCaption = document.querySelector("#editCfgScaleCaption");
const editCfgScaleSpeaker = document.querySelector("#editCfgScaleSpeaker");
const editReferenceFile = document.querySelector("#editReferenceFile");
const editReferenceChoose = document.querySelector("#editReferenceChoose");
const editReferenceDrop = document.querySelector("#editReferenceDrop");
const editReferenceStatus = document.querySelector("#editReferenceStatus");
const editExpressionSelect = document.querySelector("#editExpressionSelect");
const expressionSlotDescription = document.querySelector("#expressionSlotDescription");
const newExpressionName = document.querySelector("#newExpressionName");
const addExpressionSlot = document.querySelector("#addExpressionSlot");
const expressionImageFile = document.querySelector("#expressionImageFile");
const expressionImageChoose = document.querySelector("#expressionImageChoose");
const expressionImageDrop = document.querySelector("#expressionImageDrop");
const expressionImageStatus = document.querySelector("#expressionImageStatus");
const expressionThumbs = document.querySelector("#expressionThumbs");
const emojiStyleSelect = document.querySelector("#emojiStyle");
const emojiCustom = document.querySelector("#emojiCustom");
const saveSessionButton = document.querySelector("#saveSession");
const loadSessionButton = document.querySelector("#loadSession");
const clearContextButton = document.querySelector("#clearContext");
const sessionStatus = document.querySelector("#sessionStatus");
const lmStatus = document.querySelector("#lmStatus");
const irodoriStatus = document.querySelector("#irodoriStatus");
const speaking = document.querySelector("#speaking");
const secondSpeaking = document.querySelector("#secondSpeaking");
const DEFAULT_MAIN_CHARACTER_NAME = "リノン";
const DEFAULT_SECOND_CHARACTER_NAME = "ルヴィア";
// CFG Scale の初期値（従来 rinon が Irodori へ渡していた固定値）。
const DEFAULT_CFG_SCALE_TEXT = 3.0;
const DEFAULT_CFG_SCALE_CAPTION = 4.0;
const DEFAULT_CFG_SCALE_SPEAKER = 5.0;

// CFG Scale を数値へ正規化し 0〜20 にクランプ。数値化不可/NaN は default。
function cfgScaleOrDefault(value, fallback) {
  if (value === "" || value === null || value === undefined) return fallback;
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  return Math.max(0, Math.min(20, num));
}

// Num Steps のキャラ別既定（低ステップだと音色がリファレンスへ寄り切らないため 40）。
const DEFAULT_CHARACTER_STEPS = 40;

// Num Steps を整数へ正規化し 1〜120 にクランプ。数値化不可は fallback。
function stepsOrDefault(value, fallback) {
  if (value === "" || value === null || value === undefined) return fallback;
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  return Math.max(1, Math.min(120, Math.round(num)));
}
let mainCharacterName = DEFAULT_MAIN_CHARACTER_NAME;
let secondCharacterName = DEFAULT_SECOND_CHARACTER_NAME;
let mainReferencePath = "";
let secondReferencePath = "";
let characters = {};
let activeMainCharacterId = "rinon";
let activeSecondCharacterId = "luvia";
let editingCharacterId = "rinon";

// 現在アクティブなメインキャラの作業用会話ログ。切替/保存の前に characterHistories へ退避する。
const history = [];
// メインキャラ ID をキーにしたキャラ別会話ログの退避先。
const characterHistories = {};
// メインキャラ ID をキーにしたキャラ別 context 上限値（context 欄）の退避先。
const characterContextLimits = {};
// キャラ別値が未設定のときに使う context 上限の既定値（環境/旧設定から更新）。
let defaultContextLimit = 8200;
let queue = [];
let interactionLocked = false;
let autoMode = false;
let autoPending = false;
// 発話の識別はキャラ名ではなくスロット（"main"=1P / "second"=2P）で行う。
// 1P と 2P に同名キャラを割り当てても取り違えないようにするため。
let autoNextSlot = "main";
let playbackSlot = "main";
let lastAssistantSpeaker = "";
let lastAssistantText = "";
// 現在選択中の返答テキストの音声 URL と、全再生に適用する再生速度。
let selectedAudioUrl = "";
// 現在選択中の返答テキスト枠（再生成・削除の対象）。未選択時は null。
let selectedMessageNode = null;
// irodori 再生成の実行中フラグ（多重実行防止＆ボタン制御）。
let regenBusy = false;
let preferredPlaybackRate = 1;
let autoTopic = "";
let autoTopicQueue = [];
let autoWebContext = "";
let autoWebQuery = "";
let autoWebResults = [];
let autoTurnCount = 0;
let autoNoDialogue = false;
let externalSpeakLastId = 0;
let externalSpeakPolling = false;
let lastContextStats = null;
let audioContext = null;
let audioSource = null;
let stereoPanner = null;
let audioUnlocked = false;
let messageInputComposing = false;
let expressionImages = {
  neutral: ["/expressions/neutral.png"],
  happy: ["/expressions/happy.png"],
  surprised: ["/expressions/surprised.png"],
  soft: ["/expressions/soft.png"],
  angry: ["/expressions/angry.png"],
  worried: ["/expressions/worried.png"],
  sad: ["/expressions/sad.png"],
  shy: ["/expressions/shy.png"],
  narration: ["/expressions/narration.png"],
  fast: ["/expressions/fast.png"],
  sleepy: ["/expressions/sleepy.png"],
  phone: ["/expressions/phone.png"],
  echo: ["/expressions/echo.png"],
  muffled: ["/expressions/muffled.png"],
  throat: ["/expressions/throat.png"],
  strong: ["/expressions/strong.png"],
  teasing: ["/expressions/teasing.png"],
  pleading: ["/expressions/pleading.png"],
  exasperated: ["/expressions/exasperated.png"],
  smug: ["/expressions/smug.png"],
  sigh: ["/expressions/sigh.png"],
  gasp: ["/expressions/gasp.png"],
  breathless: ["/expressions/breathless.png"],
  yawn: ["/expressions/yawn.png"],
  humming: ["/expressions/humming.png"],
  swallow: ["/expressions/swallow.png"],
  cough: ["/expressions/cough.png"],
  sniff: ["/expressions/sniff.png"],
  pause: ["/expressions/pause.png"],
  question: ["/expressions/question.png"],
  tender: ["/expressions/tender.png"],
  broadcast: ["/expressions/broadcast.png"],
};
const expressionSlotInfo = {
  neutral: { emoji: "・", label: "通常", description: "基準になる表情" },
  happy: { emoji: "😊", label: "楽しげ", description: "嬉しそう、明るい反応" },
  surprised: { emoji: "😲", label: "驚き", description: "驚く、息をのむ" },
  soft: { emoji: "👂", label: "囁き", description: "近い距離のやわらかい声" },
  angry: { emoji: "😠", label: "怒り", description: "不満げ、拗ねる" },
  worried: { emoji: "😟", label: "心配", description: "不安そう、弱い声" },
  sad: { emoji: "😭", label: "泣き声", description: "悲しみ、嗚咽" },
  shy: { emoji: "🫣", label: "照れ", description: "恥ずかしそう" },
  narration: { emoji: "📖", label: "朗読", description: "ナレーション調" },
  fast: { emoji: "⏩", label: "早口", description: "急いで一気に話す" },
  sleepy: { emoji: "😪", label: "眠そう", description: "気だるげ、眠い声" },
  phone: { emoji: "📞", label: "電話越し", description: "スピーカー越しの質感" },
  echo: { emoji: "📢", label: "エコー", description: "響き、リバーブ" },
  broadcast: { emoji: "📢", label: "エコー", description: "放送・響きのある声" },
  muffled: { emoji: "🤐", label: "口を塞ぐ", description: "こもった声、口元の音" },
  throat: { emoji: "😖", label: "苦しげ", description: "喉に力が入る、詰まる感じ" },
  strong: { emoji: "💪", label: "力強く", description: "勢い、強い声" },
  teasing: { emoji: "😏", label: "からかう", description: "甘えるように、挑発的に" },
  pleading: { emoji: "🙏", label: "懇願", description: "お願いするように" },
  exasperated: { emoji: "🙄", label: "呆れ", description: "呆れた、困った反応" },
  smug: { emoji: "😎", label: "得意げ", description: "自信ありげ、余裕" },
  sigh: { emoji: "😮‍💨", label: "吐息", description: "溜息、息を漏らす" },
  gasp: { emoji: "😮", label: "息をのむ", description: "驚きの短い吸気" },
  breathless: { emoji: "🌬️", label: "息切れ", description: "荒い息遣い、呼吸音" },
  yawn: { emoji: "🥱", label: "あくび", description: "眠そうなあくび" },
  humming: { emoji: "🎵", label: "鼻歌", description: "ハミング、軽い歌声" },
  swallow: { emoji: "🥤", label: "飲み込む", description: "唾を飲む音" },
  cough: { emoji: "🤧", label: "咳・鼻", description: "咳き込み、鼻すすり" },
  sniff: { emoji: "👃", label: "嗅ぐ音", description: "匂いを嗅ぐ、鼻の音" },
  pause: { emoji: "⏸️", label: "間", description: "沈黙、間を置く" },
  question: { emoji: "🤔", label: "疑問", description: "考え込む、疑問形" },
  tender: { emoji: "🫶", label: "優しく", description: "Tenderly、包むように" },
};

function expressionSlotLabel(key) {
  const info = expressionSlotInfo[key];
  return info ? `${info.emoji} ${info.label} / ${key}` : `・ カスタム / ${key}`;
}

function expressionSlotDetail(key) {
  const info = expressionSlotInfo[key];
  return info ? `${info.emoji} ${info.label}: ${info.description}` : `・ カスタム: ${key}`;
}

const secondExpressionImages = {
  neutral: ["/second_player/expressions/luvia_neutral.png"],
  happy: ["/second_player/expressions/luvia_happy.png", "/second_player/expressions/luvia_laughing.png"],
  surprised: ["/second_player/expressions/luvia_surprised.png"],
  soft: ["/second_player/expressions/luvia_soft.png"],
  angry: ["/second_player/expressions/luvia_angry.png"],
  worried: ["/second_player/expressions/luvia_worried.png"],
  sad: ["/second_player/expressions/luvia_sad.png"],
  shy: [
    "/second_player/expressions/luvia_shy.png",
    "/second_player/expressions/luvia_shy_02.png",
    "/second_player/expressions/luvia_shy_03.png",
    "/second_player/expressions/luvia_shy_04.png",
    "/second_player/expressions/luvia_shy_05.png",
  ],
  strong: ["/second_player/expressions/luvia_determined.png"],
  teasing: [
    "/second_player/expressions/luvia_teasing.png",
    "/second_player/expressions/luvia_teasing_02.png",
    "/second_player/expressions/luvia_teasing_03.png",
    "/second_player/expressions/luvia_teasing_04.png",
    "/second_player/expressions/luvia_teasing_05.png",
  ],
  pleading: ["/second_player/expressions/luvia_worried.png"],
  exasperated: ["/second_player/expressions/luvia_exasperated.png"],
  smug: ["/second_player/expressions/luvia_smug.png"],
  sigh: ["/second_player/expressions/luvia_exasperated.png"],
  question: ["/second_player/expressions/luvia_question.png"],
  tender: [
    "/second_player/expressions/luvia_tender.png",
    "/second_player/expressions/luvia_tender_02.png",
    "/second_player/expressions/luvia_tender_03.png",
    "/second_player/expressions/luvia_tender_04.png",
    "/second_player/expressions/luvia_tender_05.png",
  ],
  narration: ["/second_player/expressions/luvia_serious.png"],
  broadcast: ["/second_player/expressions/luvia_serious.png"],
  fast: ["/second_player/expressions/luvia_determined.png"],
  sleepy: ["/second_player/expressions/luvia_exasperated.png"],
  phone: ["/second_player/expressions/luvia_thoughtful.png"],
  echo: ["/second_player/expressions/luvia_serious.png"],
  muffled: [
    "/second_player/expressions/luvia_muffled.png",
    "/second_player/expressions/luvia_muffled_02.png",
    "/second_player/expressions/luvia_muffled_03.png",
    "/second_player/expressions/luvia_muffled_04.png",
    "/second_player/expressions/luvia_muffled_05.png",
  ],
  throat: ["/second_player/expressions/luvia_exasperated.png"],
  gasp: ["/second_player/expressions/luvia_surprised.png"],
  breathless: ["/second_player/expressions/luvia_exasperated.png"],
  yawn: ["/second_player/expressions/luvia_exasperated.png"],
  humming: ["/second_player/expressions/luvia_happy.png"],
  swallow: ["/second_player/expressions/luvia_thoughtful.png"],
  cough: ["/second_player/expressions/luvia_worried.png"],
  sniff: ["/second_player/expressions/luvia_worried.png"],
  pause: ["/second_player/expressions/luvia_thoughtful.png"],
  thoughtful: ["/second_player/expressions/luvia_thoughtful.png"],
  laughing: ["/second_player/expressions/luvia_laughing.png"],
  serious: ["/second_player/expressions/luvia_serious.png"],
  determined: ["/second_player/expressions/luvia_determined.png"],
};

function characterById(id) {
  return characters[id] || null;
}

function expressionValuesForCharacter(character, expression) {
  const expressions = character?.expressions || {};
  const values = expressions[expression] || expressions.neutral || [];
  return Array.isArray(values) ? values.filter(Boolean) : [values].filter(Boolean);
}

function randomExpressionImage(character, expression, fallback) {
  const values = expressionValuesForCharacter(character, expression);
  const src = values[Math.floor(Math.random() * values.length)] || character?.portrait || fallback;
  return src || fallback;
}

function setExpression(name) {
  portrait.src = randomExpressionImage(characterById(activeMainCharacterId), name, "/expressions/neutral.png");
}

function setSecondExpression(name) {
  secondPortrait.src = randomExpressionImage(
    characterById(activeSecondCharacterId),
    name,
    "/Character/luvia/expressions/neutral/luvia_neutral.png"
  );
}

// スロットに対応するキャラ名を返す（表示・バックエンド送信・プロンプト文面用）。
function characterNameForSlot(slot) {
  return slot === "second" ? secondCharacterName : mainCharacterName;
}

// もう一方のスロットを返す（自動会話のターン交代・相手参照用）。
function otherSlot(slot) {
  return slot === "second" ? "main" : "second";
}

// 履歴テキストの話者名からスロットを推定する（再生成のフォールバック専用）。
// 1P/2P 同名だと名前だけでは判別できないため、その場合は 1P を既定にする。
function slotForSpeakerName(name) {
  const trimmed = String(name || "").trim();
  if (!trimmed) return defaultTurnSlot();
  if (trimmed === secondCharacterName && secondCharacterName !== mainCharacterName) return "second";
  return "main";
}

// 引数なしで発話する既定スロット。2Pキャラモードのときは 2P、それ以外は 1P。
function defaultTurnSlot() {
  return twoPlayerMode.checked ? "second" : "main";
}

function activeStage(slot = defaultTurnSlot()) {
  const isSecond = slot === "second";
  const mainCharacter = characterById(activeMainCharacterId);
  const secondCharacter = characterById(activeSecondCharacterId);
  const character = isSecond ? secondCharacter : mainCharacter;
  const fallbackName = isSecond ? secondCharacterName : mainCharacterName;
  return {
    speaker: character?.name || fallbackName,
    // RAG 長期記憶をキャラクター単位で分離するための安定した ID。
    id: character?.id || (isSecond ? activeSecondCharacterId : activeMainCharacterId),
    slot: isSecond ? "second" : "main",
    systemPrompt: character?.systemPrompt || (isSecond ? secondSystemPrompt.value : systemPrompt.value),
    ttsCaption: character?.ttsCaption || (isSecond ? secondTtsCaption.value : ttsCaption.value),
    styleGuide: character?.styleGuide || "",
    steps: stepsOrDefault(character?.steps, stepsOrDefault(stepsInput.value, DEFAULT_CHARACTER_STEPS)),
    referencePath: character?.referencePath || (isSecond ? secondReferencePath : mainReferencePath),
    cfgScaleText: cfgScaleOrDefault(character?.cfgScaleText, DEFAULT_CFG_SCALE_TEXT),
    cfgScaleCaption: cfgScaleOrDefault(character?.cfgScaleCaption, DEFAULT_CFG_SCALE_CAPTION),
    cfgScaleSpeaker: cfgScaleOrDefault(character?.cfgScaleSpeaker, DEFAULT_CFG_SCALE_SPEAKER),
  };
}

function setActiveSpeaker(slot, preserveTwoPlayer = false) {
  if (slot === "second") {
    twoPlayerMode.checked = true;
  } else if (!preserveTwoPlayer) {
    twoPlayerMode.checked = false;
  }
  updateTwoPlayerMode();
}

function setStageStatus(text, slot = defaultTurnSlot()) {
  const isSecond = slot === "second";
  speaking.textContent = isSecond ? "ready" : text;
  secondSpeaking.textContent = isSecond ? text : "standby";
}

function setSpeakingState(active, slot = defaultTurnSlot()) {
  const isSecond = slot === "second";
  document.body.classList.toggle("speaking-main", active && !isSecond);
  document.body.classList.toggle("speaking-second", active && isSecond);
  portraitWrap.classList.toggle("is-speaking", active && !isSecond);
  secondPortraitWrap.classList.toggle("is-speaking", active && isSecond);
}

function addMessage(role, text, meta = "", options = {}) {
  const node = document.createElement("div");
  node.className = `msg ${role}`;
  node.textContent = text;
  if (meta) {
    const metaNode = document.createElement("div");
    metaNode.className = "meta";
    metaNode.textContent = meta;
    node.appendChild(metaNode);
  }
  const audioUrl = options.audioUrl || "";
  if (role === "assistant" && audioUrl) {
    node.classList.add("has-audio");
    node.dataset.audioUrl = audioUrl;
    // 再生成に必要な生成時パラメータ（感情セグメント・キャラ別CFG/steps/reference等）を枠に紐づける。
    // リロード後も再生成できるよう history.display.regen へも保存する（renderHistory で復元）。
    if (options.regen && typeof options.regen === "object") {
      node._regen = options.regen;
    }
    node.addEventListener("click", () => selectReplyMessage(node));
  }
  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return node;
}

// 選択状態に応じて再生成・削除ボタンの活性/非活性を切り替える。
// 再生成は音声付きの選択枠なら有効（生成時パラメータが無い過去の会話でも、履歴テキストと
// 現在のキャラ設定からフォールバックで組み立てて再生成できる）。削除は選択枠なら常に有効。
function updateSelectionActions() {
  const hasSelection = Boolean(selectedMessageNode);
  const hasAudio =
    hasSelection && Boolean(selectedMessageNode.dataset && selectedMessageNode.dataset.audioUrl);
  if (regenAudioButton) regenAudioButton.disabled = !hasAudio || regenBusy;
  if (deleteMessageButton) deleteMessageButton.disabled = !hasSelection;
}

// 現在選択中の返答テキスト枠をハイライトする（他の枠の選択は解除）。
function highlightSelectedMessage(node) {
  messagesEl
    .querySelectorAll(".msg.assistant.selected")
    .forEach((el) => el.classList.remove("selected"));
  if (node) node.classList.add("selected");
  selectedMessageNode = node || null;
  updateSelectionActions();
}

// 全再生に適用する再生速度を #player に反映する。新しい音源を読み込んでも維持されるよう
// defaultPlaybackRate と playbackRate の両方を設定する。
function applyPreferredPlaybackRate() {
  try {
    player.defaultPlaybackRate = preferredPlaybackRate;
    player.playbackRate = preferredPlaybackRate;
  } catch (_) {
    /* 一部ブラウザで読み込み前に設定すると例外になることがあるため握りつぶす。 */
  }
}

// 指定した音声を #player に読み込む。▶再生・ダウンロード・保存はすべて #player を対象に
// するため、「選択中の返答」を切り替える＝player.src を差し替えること。
function loadAudioIntoPlayer(url, { autoplay = false } = {}) {
  if (!url) return;
  selectedAudioUrl = url;
  player.src = url;
  applyPreferredPlaybackRate();
  audioSaveStatus.textContent = "save ready";
  if (autoplay) {
    player.play().catch(() => {});
  }
}

// 返答枠クリック時の選択。読み込みのみ行い、再生はユーザー操作の▶に任せる。
function selectReplyMessage(node) {
  const url = node && node.dataset ? node.dataset.audioUrl : "";
  if (!url) return;
  highlightSelectedMessage(node);
  loadAudioIntoPlayer(url, { autoplay: false });
}

// 音声 URL から対応する返答枠を探す（自動再生時のハイライト同期用）。
function findMessageByAudioUrl(url) {
  if (!url) return null;
  const nodes = messagesEl.querySelectorAll(".msg.assistant.has-audio");
  for (const el of nodes) {
    if (el.dataset.audioUrl === url) return el;
  }
  return null;
}

// meta 行の「style」部分を組み立てる。
// 感情セグメントが複数、または感情ラベル付きなら「絵文字+短い日本語タグ」を→で並べる（案1）。
// セグメント情報が無い/単一で無感情なら、従来どおり絵文字1つを表示する。
function buildStyleMeta(data) {
  const label = data.autoEmoji && data.llmEmojiStyle ? "auto style" : "style";
  const segments = Array.isArray(data.segments) ? data.segments : [];
  const shorten = (value) => {
    const text = String(value || "").trim();
    return text.length > 8 ? `${text.slice(0, 8)}…` : text;
  };
  const hasEmotion = segments.some((seg) => (seg.style || "").trim());
  if (segments.length > 1 || (segments.length === 1 && hasEmotion)) {
    const maxShown = 6;
    const parts = segments
      .map((seg) => `${(seg.emoji || "").trim()}${shorten(seg.style)}`.trim())
      .filter(Boolean);
    if (!parts.length) return "";
    const shown = parts.slice(0, maxShown);
    const suffix = parts.length > maxShown ? "…" : "";
    return ` / ${label} ${shown.join("→")}${suffix}`;
  }
  return data.emojiStyle ? ` / ${label} ${data.emojiStyle}` : "";
}

// irodori 再生成に必要な「生成時パラメータ一式」を組み立てる。返答テキスト・感情セグメント・
// キャラ別 CFG/steps/reference をそのまま保持し、辞書やコード修正のみを合成時に反映させて
// 同じ返答を作り直せるようにする。リモート TTS 設定は再生成時に UI から都度読む。
function buildRegenPayload(data, stage) {
  return {
    reply: data.reply || "",
    segments: Array.isArray(data.segments) ? data.segments : [],
    emojiStyle: data.emojiStyle || "",
    llmEmojiStyle: data.llmEmojiStyle || "",
    speaker: data.speaker || stage.speaker,
    speakerSlot: stage.slot,
    ttsCaption: stage.ttsCaption,
    cfgScaleText: stage.cfgScaleText,
    cfgScaleCaption: stage.cfgScaleCaption,
    cfgScaleSpeaker: stage.cfgScaleSpeaker,
    steps: stage.steps,
    referencePath: stage.slot === "main" ? stage.referencePath : mainReferencePath,
    secondReferencePath: stage.slot === "second" ? stage.referencePath : secondReferencePath,
    speechRate: data.speechRate || "normal",
  };
}

// 表示用の返答テキストに、各セグメント先頭へ「（絵文字＋感情キャプション全文）」を差し込む。
// 感情キャプションは省略せずそのまま挿入する。履歴・音声用の data.reply は変更しない。
function buildAnnotatedReply(data) {
  const segments = Array.isArray(data.segments) ? data.segments : [];
  const haveTexts = segments.length > 0 && segments.every((seg) => typeof seg.text === "string");
  const hasMarker = segments.some((seg) => (seg.style || "").trim() || (seg.emoji || "").trim());
  if (!haveTexts || !hasMarker) return data.reply;
  return segments
    .map((seg) => {
      const marker = `${(seg.emoji || "").trim()}${(seg.style || "").trim()}`;
      const text = seg.text || "";
      return marker ? `（${marker}）${text}` : text;
    })
    .join("");
}

function contextCost(items) {
  return items.reduce((total, item) => total + String(item.content || "").length + 32, 0);
}

function updateContextUsage(extraText = "") {
  const limit = Number(contextLimit.value || 8200);
  const pending = extraText ? [{ role: "user", content: extraText }] : [];
  const used = contextCost([...history, ...pending]);
  const compactLabel = lastContextStats
    ? ` · LM ${lastContextStats.sent}/${lastContextStats.effectiveLimit}${lastContextStats.compacted ? " compact" : ""}`
    : "";
  contextUsage.textContent = `${used} / ${limit}${compactLabel}`;
  contextUsage.classList.toggle("is-near", used > limit * 0.8);
  contextUsage.classList.toggle("is-over", used > limit);
}

function setInteractionLocked(locked, label = "Send") {
  interactionLocked = locked;
  sendButton.disabled = locked && !autoMode;
  sendButton.textContent = autoMode ? "Cue" : label;
  messageInput.readOnly = false;
  updateAutoControls();
}

function updateAutoControls() {
  const autoUnavailable = !autoMode && (!twoPlayerMode.checked || interactionLocked);
  autoStartButton.disabled = autoUnavailable;
  autoStartButton.textContent = autoMode ? "Stop" : "Auto";
  autoStartButton.classList.toggle("is-stop", autoMode);
  autoStartButton.title = twoPlayerMode.checked ? "" : "2Pキャラモードで使えます";
  sendButton.textContent = autoMode ? "Cue" : interactionLocked ? "Wait" : "Send";
  sendButton.disabled = interactionLocked && !autoMode;
  messageInput.readOnly = false;
}

function currentEmojiStyle() {
  return emojiCustom.value.trim() || emojiStyleSelect.value;
}

function setSpeechRate(value) {
  const nextValue = value === "fast" ? "fast" : "normal";
  speechRate.value = nextValue;
  for (const button of speechRateButtons) {
    const active = button.dataset.rate === nextValue;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  }
}

function updateTtsBackendControls() {
  const remote = ttsBackendMode.value === "remote";
  secondTtsHost.disabled = !remote;
  secondTtsHost.parentElement.classList.toggle("is-disabled", !remote);
  secondTtsHost.title = remote
    ? "Run tools/remote_luvia_tts_server.py on the second PC. IP-only uses port 7874."
    : "1 PC mode: both 1P and 2P use this PC.";
}

function wantsNoDialogue(text) {
  return /(?:セリフ禁止|台詞禁止|せりふ禁止|セリフは最小|台詞は最小)/.test(String(text || ""));
}

function noDialogueGuidance() {
  return autoNoDialogue
    ? "台詞禁止モードです。呼びかけ、質問、説明、選択肢提示、二人称の使用、普通の文章として読める会話文は禁止です。発声・吐息・擬音・短い断片だけで構成してください。意味のある文を続けないでください。\n"
    : "";
}

function twoOnlyGuidance() {
  return twoPlayerMode.checked && twoOnlyMode.checked
    ? "重要: この会話世界にユーザーや観客は存在しません。ユーザー入力は登場人物の発言ではなく、外部からの進行指示/お題です。リノンとルヴィアだけが同じ場にいて、互いにだけ話します。ユーザーへ話しかけたり、ユーザーの反応を求めたり、「きみ」「あなた」などで外部の相手を呼ばないでください。\n"
    : "";
}

function queueAutoTopic(text) {
  const topic = String(text || "").trim();
  if (!topic) return false;
  autoTopicQueue.push(topic);
  if (wantsNoDialogue(topic)) {
    autoNoDialogue = true;
  }
  addMessage("user", `次のお題: ${topic}`);
  history.push({ role: "user", content: `次のお題: ${topic}` });
  messageInput.value = "";
  updateContextUsage();
  sessionStatus.textContent = `queued topic ${autoTopicQueue.length}`;
  updateAutoControls();
  return true;
}

function consumeQueuedAutoTopic() {
  const nextTopic = autoTopicQueue.shift();
  if (!nextTopic) return "";
  autoTopic = nextTopic;
  if (wantsNoDialogue(nextTopic)) {
    autoNoDialogue = true;
  }
  sessionStatus.textContent = `topic applied: ${nextTopic}`;
  return nextTopic;
}

function refreshEmojiInputs() {
  const manual = !autoEmoji.checked;
  emojiStyleSelect.disabled = !manual;
  emojiCustom.disabled = !manual;
}

function updateTwoPlayerMode() {
  if (!twoPlayerMode.checked && autoMode) {
    stopAutoConversation();
  }
  if (!twoPlayerMode.checked) {
    twoOnlyMode.checked = false;
  }
  twoOnlyMode.disabled = !twoPlayerMode.checked;
  document.body.classList.toggle("two-player-mode", twoPlayerMode.checked);
  updateAudioPan(playbackSlot);
  if (!interactionLocked) {
    speaking.textContent = "ready";
    secondSpeaking.textContent = twoPlayerMode.checked ? "ready" : "standby";
    setSpeakingState(false);
  }
}

function ensureAudioPanner() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass || !AudioContextClass.prototype.createStereoPanner) {
    return false;
  }
  if (!audioContext) {
    audioContext = new AudioContextClass();
    audioSource = audioContext.createMediaElementSource(player);
    stereoPanner = audioContext.createStereoPanner();
    audioSource.connect(stereoPanner).connect(audioContext.destination);
  }
  if (audioContext.state === "suspended") {
    audioContext.resume().catch(() => {});
  }
  return Boolean(stereoPanner);
}

function unlockAudioPlayback() {
  if (audioUnlocked) return;
  audioUnlocked = true;
  ensureAudioPanner();
  if (player.paused) {
    player.play().catch(() => {});
  }
}

function updateAudioPan(slot = defaultTurnSlot()) {
  if (!ensureAudioPanner()) return;
  if (!twoPlayerMode.checked) {
    stereoPanner.pan.value = 0;
    return;
  }
  stereoPanner.pan.value = slot === "second" ? 0.22 : -0.22;
}

function setSelectValue(select, value) {
  if (!value) return;
  const exists = [...select.options].some((option) => option.value === value);
  if (!exists) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }
  select.value = value;
}

function normalizeSendShortcut(value) {
  if (value === "meta-enter" || value === "ctrl-enter") return value;
  return "enter";
}

function eventMatchesSendShortcut(event) {
  if (event.key !== "Enter") return false;
  if (event.isComposing || messageInputComposing || event.keyCode === 229) return false;
  const shortcut = normalizeSendShortcut(sendShortcut.value);
  if (shortcut === "meta-enter") {
    return event.metaKey && !event.ctrlKey && !event.altKey && !event.shiftKey;
  }
  if (shortcut === "ctrl-enter") {
    return event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey;
  }
  return !event.shiftKey && !event.metaKey && !event.ctrlKey && !event.altKey;
}

function shortPathLabel(path) {
  const text = String(path || "").trim();
  if (!text) return "default reference";
  const normalized = text.replaceAll("\\", "/");
  return normalized.split("/").pop() || text;
}

function characterList() {
  return Object.values(characters).sort((a, b) => String(a.name || "").localeCompare(String(b.name || ""), "ja"));
}

function expressionKeys(character) {
  const keys = Object.keys(character?.expressions || {});
  if (!keys.includes("neutral")) keys.unshift("neutral");
  return [...new Set(keys)].sort((a, b) => (a === "neutral" ? -1 : b === "neutral" ? 1 : a.localeCompare(b)));
}

function syncActiveCharacterState() {
  const mainCharacter = characterById(activeMainCharacterId) || characterList()[0];
  const secondCharacter = characterById(activeSecondCharacterId) || characterList()[1] || mainCharacter;
  if (mainCharacter) activeMainCharacterId = mainCharacter.id;
  if (secondCharacter) activeSecondCharacterId = secondCharacter.id;
  mainCharacterName = mainCharacter?.name || DEFAULT_MAIN_CHARACTER_NAME;
  secondCharacterName = secondCharacter?.name || DEFAULT_SECOND_CHARACTER_NAME;
  mainReferencePath = mainCharacter?.referencePath || mainReferencePath;
  secondReferencePath = secondCharacter?.referencePath || secondReferencePath;
  // 1P と 2P が同名だと左右の区別がつかないため、その場合だけ枠を併記する。
  const sameName = mainCharacterName === secondCharacterName;
  mainCharacterNameLabel.textContent = sameName ? `${mainCharacterName}（1P）` : mainCharacterName;
  secondCharacterNameLabel.textContent = sameName ? `${secondCharacterName}（2P）` : secondCharacterName;
  systemPrompt.value = mainCharacter?.systemPrompt || systemPrompt.value;
  ttsCaption.value = mainCharacter?.ttsCaption || ttsCaption.value;
  secondSystemPrompt.value = secondCharacter?.systemPrompt || secondSystemPrompt.value;
  secondTtsCaption.value = secondCharacter?.ttsCaption || secondTtsCaption.value;
  setExpression("neutral");
  setSecondExpression("neutral");
  updateTwoPlayerMode();
}

function populateCharacterSelect(select, value) {
  select.innerHTML = "";
  for (const character of characterList()) {
    const option = document.createElement("option");
    option.value = character.id;
    option.textContent = character.name || character.id;
    select.appendChild(option);
  }
  if (value && characters[value]) select.value = value;
}

function refreshCharacterSelectors() {
  populateCharacterSelect(mainCharacterSelect, activeMainCharacterId);
  populateCharacterSelect(secondCharacterSelect, activeSecondCharacterId);
  populateCharacterSelect(editCharacterSelect, editingCharacterId);
}

function renderExpressionEditor() {
  const character = characterById(editingCharacterId);
  editExpressionSelect.innerHTML = "";
  for (const key of expressionKeys(character)) {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = expressionSlotLabel(key);
    option.title = expressionSlotDetail(key);
    editExpressionSelect.appendChild(option);
  }
  if (!editExpressionSelect.value && editExpressionSelect.options.length) {
    editExpressionSelect.value = editExpressionSelect.options[0].value;
  }
  renderExpressionThumbs();
}

function renderExpressionThumbs() {
  const character = characterById(editingCharacterId);
  const key = editExpressionSelect.value || "neutral";
  const values = expressionValuesForCharacter(character, key);
  expressionThumbs.innerHTML = "";
  expressionSlotDescription.textContent = expressionSlotDetail(key);
  for (const url of values) {
    const img = document.createElement("img");
    img.src = url;
    img.alt = key;
    expressionThumbs.appendChild(img);
  }
  expressionImageStatus.textContent = values.length ? `${values.length} images` : "no image selected";
}

function renderCharacterEditor() {
  const character = characterById(editingCharacterId);
  if (!character) return;
  editCharacterName.value = character.name || "";
  editSystemPrompt.value = character.systemPrompt || "";
  editTtsCaption.value = character.ttsCaption || "";
  editStyleGuide.value = character.styleGuide || "";
  editSteps.value = stepsOrDefault(character.steps, DEFAULT_CHARACTER_STEPS);
  editCfgScaleText.value = cfgScaleOrDefault(character.cfgScaleText, DEFAULT_CFG_SCALE_TEXT);
  editCfgScaleCaption.value = cfgScaleOrDefault(character.cfgScaleCaption, DEFAULT_CFG_SCALE_CAPTION);
  editCfgScaleSpeaker.value = cfgScaleOrDefault(character.cfgScaleSpeaker, DEFAULT_CFG_SCALE_SPEAKER);
  editReferenceStatus.textContent = shortPathLabel(character.referencePath);
  renderExpressionEditor();
}

function openOptions() {
  refreshCharacterSelectors();
  renderCharacterEditor();
  optionsModal.hidden = false;
  editCharacterName.focus();
}

function closeOptions() {
  syncActiveCharacterState();
  optionsModal.hidden = true;
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("file read failed"));
    reader.readAsDataURL(file);
  });
}

async function uploadReferenceFile(slot, file) {
  if (!file) return;
  const status = editReferenceStatus;
  status.textContent = "uploading...";
  const dataBase64 = await fileToDataUrl(file);
  const res = await fetch("/api/reference", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      slot,
      characterId: editingCharacterId,
      name: file.name,
      type: file.type,
      dataBase64,
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  const character = characterById(editingCharacterId);
  if (character) {
    character.referencePath = data.path;
  }
  renderCharacterEditor();
  syncActiveCharacterState();
  sessionStatus.textContent = `${character?.name || "character"} reference loaded`;
}

function installReferenceDrop(drop, input, chooseButton, slot) {
  chooseButton.addEventListener("click", () => input.click());
  input.addEventListener("change", async () => {
    try {
      await uploadReferenceFile(slot, input.files?.[0]);
    } catch (error) {
      sessionStatus.textContent = `reference error: ${error.message}`;
    } finally {
      input.value = "";
    }
  });
  for (const eventName of ["dragenter", "dragover"]) {
    drop.addEventListener(eventName, (event) => {
      event.preventDefault();
      drop.classList.add("is-dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    drop.addEventListener(eventName, (event) => {
      event.preventDefault();
      drop.classList.remove("is-dragging");
    });
  }
  drop.addEventListener("drop", async (event) => {
    try {
      await uploadReferenceFile(slot, event.dataTransfer?.files?.[0]);
    } catch (error) {
      sessionStatus.textContent = `reference error: ${error.message}`;
    }
  });
}

function makeNewCharacter() {
  const id = `character_${Date.now().toString(36)}`;
  characters[id] = {
    id,
    name: "New Character",
    systemPrompt: "",
    ttsCaption: ttsCaption.value || "",
    styleGuide: "",
    steps: DEFAULT_CHARACTER_STEPS,
    cfgScaleText: DEFAULT_CFG_SCALE_TEXT,
    cfgScaleCaption: DEFAULT_CFG_SCALE_CAPTION,
    cfgScaleSpeaker: DEFAULT_CFG_SCALE_SPEAKER,
    referencePath: mainReferencePath || "",
    portrait: "/expressions/neutral.png",
    expressions: {
      neutral: ["/expressions/neutral.png"],
    },
  };
  editingCharacterId = id;
  refreshCharacterSelectors();
  renderCharacterEditor();
}

function updateEditingCharacter() {
  const character = characterById(editingCharacterId);
  if (!character) return;
  character.name = editCharacterName.value.trim() || character.id;
  character.systemPrompt = editSystemPrompt.value;
  character.ttsCaption = editTtsCaption.value;
  character.styleGuide = editStyleGuide.value;
  character.steps = stepsOrDefault(editSteps.value, DEFAULT_CHARACTER_STEPS);
  character.cfgScaleText = cfgScaleOrDefault(editCfgScaleText.value, DEFAULT_CFG_SCALE_TEXT);
  character.cfgScaleCaption = cfgScaleOrDefault(editCfgScaleCaption.value, DEFAULT_CFG_SCALE_CAPTION);
  character.cfgScaleSpeaker = cfgScaleOrDefault(editCfgScaleSpeaker.value, DEFAULT_CFG_SCALE_SPEAKER);
  refreshCharacterSelectors();
  syncActiveCharacterState();
}

function addExpressionSlotForEditingCharacter() {
  const character = characterById(editingCharacterId);
  if (!character) return;
  const key = newExpressionName.value.trim().replace(/[^0-9A-Za-z_-]+/g, "_") || "neutral";
  character.expressions = character.expressions || {};
  character.expressions[key] = character.expressions[key] || [];
  newExpressionName.value = "";
  renderExpressionEditor();
  editExpressionSelect.value = key;
  renderExpressionThumbs();
}

async function uploadExpressionImage(file) {
  const character = characterById(editingCharacterId);
  if (!character || !file) return;
  const expression = editExpressionSelect.value || "neutral";
  expressionImageStatus.textContent = "uploading...";
  const dataBase64 = await fileToDataUrl(file);
  const res = await fetch("/api/character-image", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      characterId: character.id,
      expression,
      name: file.name,
      type: file.type,
      dataBase64,
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  character.expressions = character.expressions || {};
  character.expressions[expression] = character.expressions[expression] || [];
  character.expressions[expression].push(data.url);
  if (expression === "neutral" || !character.portrait) {
    character.portrait = data.url;
  }
  renderExpressionEditor();
  syncActiveCharacterState();
}

async function uploadExpressionImages(files) {
  const list = [...(files || [])];
  if (!list.length) return;
  for (const file of list) {
    await uploadExpressionImage(file);
  }
  sessionStatus.textContent = `added ${list.length} expression image${list.length > 1 ? "s" : ""}`;
}

function installImageDrop(drop, input, chooseButton) {
  chooseButton.addEventListener("click", () => input.click());
  input.addEventListener("change", async () => {
    try {
      await uploadExpressionImages(input.files);
    } catch (error) {
      sessionStatus.textContent = `image error: ${error.message}`;
    } finally {
      input.value = "";
    }
  });
  for (const eventName of ["dragenter", "dragover"]) {
    drop.addEventListener(eventName, (event) => {
      event.preventDefault();
      drop.classList.add("is-dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    drop.addEventListener(eventName, (event) => {
      event.preventDefault();
      drop.classList.remove("is-dragging");
    });
  }
  drop.addEventListener("drop", async (event) => {
    try {
      await uploadExpressionImages(event.dataTransfer?.files);
    } catch (error) {
      sessionStatus.textContent = `image error: ${error.message}`;
    }
  });
}

function characterPayload() {
  return {
    version: 1,
    activeMainId: activeMainCharacterId,
    activeSecondId: activeSecondCharacterId,
    characters: characterList(),
  };
}

function applyCharacterProfile(profile) {
  characters = {};
  for (const character of profile.characters || []) {
    if (character?.id) {
      characters[character.id] = character;
    }
  }
  activeMainCharacterId = profile.activeMainId || Object.keys(characters)[0] || "rinon";
  activeSecondCharacterId = profile.activeSecondId || Object.keys(characters)[1] || activeMainCharacterId;
  editingCharacterId = editingCharacterId && characters[editingCharacterId] ? editingCharacterId : activeMainCharacterId;
  refreshCharacterSelectors();
  syncActiveCharacterState();
  renderCharacterEditor();
}

async function loadCharacters() {
  const res = await fetch("/api/characters");
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  applyCharacterProfile(data);
}

async function saveCharacters() {
  updateEditingCharacter();
  const res = await fetch("/api/characters", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(characterPayload()),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  applyCharacterProfile(data);
  sessionStatus.textContent = `saved ${data.characters?.length || 0} characters`;
}

// history エントリの浅いコピー（display はネストごと複製）。退避先での意図しない共有を防ぐ。
function cloneHistoryEntry(entry) {
  const copy = { role: entry.role, content: entry.content };
  if (entry.display && typeof entry.display === "object") {
    copy.display = { ...entry.display };
  }
  return copy;
}

// 現在の history を、アクティブなメインキャラのログとして characterHistories に退避する。
function commitActiveHistory() {
  characterHistories[activeMainCharacterId] = history.map(cloneHistoryEntry);
}

// context 欄の現在値（数値）。空/不正時は既定値。
function currentContextLimitValue() {
  return Number(contextLimit.value) || defaultContextLimit;
}

// 現在の context 欄の値を、アクティブなメインキャラの値として退避する。
function commitActiveContextLimit() {
  characterContextLimits[activeMainCharacterId] = currentContextLimitValue();
}

// 指定キャラの context 上限値を context 欄へ反映する（未設定なら既定値）。
function applyContextLimitForCharacter(charId) {
  contextLimit.value = characterContextLimits[charId] || defaultContextLimit;
  // 環境デフォルトで上書きされないよう touched を立てる（キャラ別管理下に置く）。
  contextLimit.dataset.touched = "1";
  updateContextUsage();
}

// メインキャラ切替時に、直前キャラのログ・context 値を退避し、新キャラの内容を画面へ復元する。
function switchCharacterHistory(previousId, nextId) {
  if (previousId) {
    characterHistories[previousId] = history.map(cloneHistoryEntry);
    characterContextLimits[previousId] = currentContextLimitValue();
  }
  // 再生中の音声とキューは会話単位でリセットする。
  autoMode = false;
  autoPending = false;
  queue = [];
  player.pause();
  player.removeAttribute("src");
  player.load();
  selectedAudioUrl = "";
  selectedMessageNode = null;
  renderHistory(characterHistories[nextId] || []);
  applyContextLimitForCharacter(nextId);
  updateSelectionActions();
  updateAutoControls();
}

function sessionPayload() {
  // 保存前に現在の会話ログと context 値をアクティブキャラの値として確定させる。
  commitActiveHistory();
  commitActiveContextLimit();
  return {
    settings: {
      systemPrompt: systemPrompt.value,
      mainCharacterName,
      secondCharacterName,
      activeMainCharacterId,
      activeSecondCharacterId,
      userAddress: userAddress.value,
      ttsCaption: ttsCaption.value,
      secondSystemPrompt: secondSystemPrompt.value,
      secondTtsCaption: secondTtsCaption.value,
      referencePath: mainReferencePath,
      secondReferencePath,
      contextLimit: Number(contextLimit.value || 8200),
      characterContextLimits,
      model: modelSelect.value,
      steps: Number(stepsInput.value || 12),
      speechRate: speechRate.value,
      replyLength: replyLength.value,
      llmGenerationMode: llmGenerationMode.value,
      sendShortcut: normalizeSendShortcut(sendShortcut.value),
      ttsBackendMode: ttsBackendMode.value,
      secondTtsHost: secondTtsHost.value.trim(),
      autoEmoji: autoEmoji.checked,
      webSearch: webSearch.checked,
      twoPlayerMode: twoPlayerMode.checked,
      twoOnlyMode: twoOnlyMode.checked,
      emojiStyle: emojiStyleSelect.value,
      emojiCustom: emojiCustom.value,
    },
    histories: characterHistories,
  };
}

function renderHistory(items) {
  history.length = 0;
  messagesEl.innerHTML = "";
  lastContextStats = null;
  let lastAudioNode = null;
  for (const item of items || []) {
    if (!item || !["user", "assistant"].includes(item.role) || !item.content) continue;
    const entry = { role: item.role, content: item.content };
    const display = item.role === "assistant" && item.display && typeof item.display === "object"
      ? item.display
      : null;
    if (display) {
      entry.display = display;
      const node = addMessage("assistant", display.text || item.content, display.meta || "", {
        audioUrl: display.audioUrl || "",
        regen: display.regen,
      });
      if (display.audioUrl) lastAudioNode = node;
    } else {
      addMessage(item.role, item.content);
    }
    history.push(entry);
  }
  // 復元直後は、最後の音声付き返答を選択状態（ハイライト＋再生対象）にする。
  if (lastAudioNode) selectReplyMessage(lastAudioNode);
  updateContextUsage();
}

function clearContext() {
  const ok = window.confirm("今の会話コンテキストを消して、ゼロからやり直します。消してもいいですか？");
  if (!ok) return;
  autoMode = false;
  autoPending = false;
  lastContextStats = null;
  autoWebContext = "";
  autoWebQuery = "";
  autoWebResults = [];
  history.length = 0;
  // アクティブなメインキャラのログのみを消去する（他キャラのログは保持）。
  characterHistories[activeMainCharacterId] = [];
  messagesEl.innerHTML = "";
  queue = [];
  player.pause();
  player.removeAttribute("src");
  player.load();
  selectedAudioUrl = "";
  selectedMessageNode = null;
  updateSelectionActions();
  audioSaveStatus.textContent = "no audio";
  speaking.textContent = "ready";
  secondSpeaking.textContent = "standby";
  setSpeakingState(false);
  setExpression("neutral");
  setInteractionLocked(false);
  updateAutoControls();
  sessionStatus.textContent = "context cleared";
  updateContextUsage();
}

// preserveActiveCharacter=true のとき、選択中のキャラ（メイン/2P）と、そのキャラ固有の
// 設定（名前・プロンプト・参照音声）は保存内容で上書きせず現状を維持する。手動 Load 用。
function applySession(profile, preserveActiveCharacter = false) {
  const settings = profile.settings || {};
  if (!preserveActiveCharacter) {
    if (settings.activeMainCharacterId && characters[settings.activeMainCharacterId]) {
      activeMainCharacterId = settings.activeMainCharacterId;
    }
    if (settings.activeSecondCharacterId && characters[settings.activeSecondCharacterId]) {
      activeSecondCharacterId = settings.activeSecondCharacterId;
    }
    mainCharacterName = settings.mainCharacterName || mainCharacterName || DEFAULT_MAIN_CHARACTER_NAME;
    secondCharacterName = settings.secondCharacterName || secondCharacterName || DEFAULT_SECOND_CHARACTER_NAME;
    mainReferencePath = settings.referencePath || mainReferencePath;
    secondReferencePath = settings.secondReferencePath || secondReferencePath;
    if (settings.systemPrompt) systemPrompt.value = settings.systemPrompt;
    if (settings.ttsCaption) ttsCaption.value = settings.ttsCaption;
    if (settings.secondSystemPrompt) secondSystemPrompt.value = settings.secondSystemPrompt;
    if (settings.secondTtsCaption) secondTtsCaption.value = settings.secondTtsCaption;
    if (secondSystemPrompt.value.startsWith("2Pは")) {
      secondSystemPrompt.value = secondSystemPrompt.value.replace(/^2Pは/, `${secondCharacterName}は`);
    }
  }
  userAddress.value = settings.userAddress || userAddress.value || "あなた";
  // 旧グローバル context 値は、キャラ別値の既定値（フォールバック）として取り込む。
  if (settings.contextLimit) defaultContextLimit = Number(settings.contextLimit) || defaultContextLimit;
  updateContextUsage();
  setSelectValue(modelSelect, settings.model);
  if (settings.steps) stepsInput.value = settings.steps;
  setSpeechRate(settings.speechRate);
  if (settings.replyLength) replyLength.value = settings.replyLength;
  if (settings.llmGenerationMode) llmGenerationMode.value = settings.llmGenerationMode;
  sendShortcut.value = normalizeSendShortcut(settings.sendShortcut);
  ttsBackendMode.value = settings.ttsBackendMode === "remote" ? "remote" : "local";
  if (Object.prototype.hasOwnProperty.call(settings, "secondTtsHost")) {
    secondTtsHost.value = settings.secondTtsHost || "";
  }
  updateTtsBackendControls();
  autoEmoji.checked = Boolean(settings.autoEmoji ?? true);
  webSearch.checked = Boolean(settings.webSearch ?? false);
  twoPlayerMode.checked = Boolean(settings.twoPlayerMode ?? false);
  twoOnlyMode.checked = Boolean(settings.twoOnlyMode ?? false);
  refreshCharacterSelectors();
  syncActiveCharacterState();
  updateTwoPlayerMode();
  setSelectValue(emojiStyleSelect, settings.emojiStyle);
  emojiCustom.value = settings.emojiCustom || "";
  refreshEmojiInputs();
  // キャラ別ログを復元。旧フォーマット（単一 history）はアクティブキャラのログへ振り分ける。
  for (const key of Object.keys(characterHistories)) delete characterHistories[key];
  const histories = profile.histories && typeof profile.histories === "object" ? profile.histories : null;
  if (histories) {
    for (const [id, items] of Object.entries(histories)) {
      if (Array.isArray(items)) characterHistories[id] = items.map(cloneHistoryEntry);
    }
  } else if (Array.isArray(profile.history) && profile.history.length) {
    characterHistories[activeMainCharacterId] = profile.history.map(cloneHistoryEntry);
  }
  renderHistory(characterHistories[activeMainCharacterId] || []);
  // キャラ別 context 上限値を復元。旧フォーマット（キャラ別マップ無し）は旧グローバル値を
  // 全既知キャラの初期値として展開する（読込時の自動振り分け）。
  for (const key of Object.keys(characterContextLimits)) delete characterContextLimits[key];
  const ctxMap = settings.characterContextLimits && typeof settings.characterContextLimits === "object"
    ? settings.characterContextLimits
    : null;
  if (ctxMap) {
    for (const [id, value] of Object.entries(ctxMap)) {
      const n = Number(value);
      if (n > 0) characterContextLimits[id] = n;
    }
  }
  if (!Object.keys(characterContextLimits).length && settings.contextLimit) {
    const base = Number(settings.contextLimit) || defaultContextLimit;
    for (const id of Object.keys(characters)) characterContextLimits[id] = base;
  }
  applyContextLimitForCharacter(activeMainCharacterId);
  const activeCount = (characterHistories[activeMainCharacterId] || []).length;
  sessionStatus.textContent = profile.savedAt ? `loaded ${activeCount} turns` : "loaded";
}

async function saveSession() {
  const res = await fetch("/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(sessionPayload()),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  sessionStatus.textContent = `saved ${data.historyCount} turns`;
}

// 会話が1ターン追加されるたびに、手動 Save と同じ保存処理を静かに実行する。
// フローを妨げないよう例外は握りつぶし、多重実行は直列化する（保存中に届いた要求は
// 完了後にもう一度だけ走らせ、最新状態を取りこぼさない）。
let autoSaveInFlight = null;
let autoSavePending = false;
function autoSaveSession() {
  if (autoSaveInFlight) {
    autoSavePending = true;
    return;
  }
  autoSaveInFlight = (async () => {
    try {
      await saveSession();
    } catch (error) {
      sessionStatus.textContent = `auto-save error: ${error.message}`;
    } finally {
      autoSaveInFlight = null;
      if (autoSavePending) {
        autoSavePending = false;
        autoSaveSession();
      }
    }
  })();
}

async function loadSession(silent = false, preserveActiveCharacter = false) {
  const res = await fetch("/api/session");
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  if (!data.exists) {
    if (!silent) sessionStatus.textContent = "no save";
    return;
  }
  applySession(data, preserveActiveCharacter);
}

function playQueue(items, slot = activeStage().slot, options = {}) {
  const nextItems = [...items].map((item) => ({ ...item, slot }));
  const isAudioActive = Boolean(player.currentSrc) && !player.paused && !player.ended;
  if (options.append && isAudioActive) {
    queue.push(...nextItems);
    return;
  }
  queue = options.append ? [...queue, ...nextItems] : nextItems;
  playbackSlot = slot;
  if (queue.length === 0) {
    setInteractionLocked(false);
    return;
  }
  playNext();
}

function hasBufferedOtherSlot(slot) {
  return queue.some((item) => (item.slot || slot) !== slot);
}

function playNext() {
  const next = queue.shift();
  if (!next) {
    speaking.textContent = "ready";
    secondSpeaking.textContent = "standby";
    setSpeakingState(false);
    setExpression("neutral");
    setSecondExpression("neutral");
    setInteractionLocked(false);
    if (autoMode && !autoPending) {
      continueAutoConversation();
    }
    return;
  }
  playbackSlot = next.slot || playbackSlot;
  updateAudioPan(playbackSlot);
  let deferredNode = null;
  if (next.deferredMessage) {
    deferredNode = addMessage(
      "assistant",
      next.deferredMessage.reply,
      next.deferredMessage.meta,
      { audioUrl: next.deferredMessage.audioUrl || next.url || "", regen: next.deferredMessage.regen },
    );
  }
  if (playbackSlot === "second") {
    setSecondExpression(next.expression || "neutral");
  } else {
    setExpression(next.expression || "neutral");
  }
  setStageStatus("speaking", playbackSlot);
  setSpeakingState(true, playbackSlot);
  // 再生する返答を「現在選択中」にしてハイライトを同期する。
  // 分割音声（結合なし）で後続チャンク再生中は一致枠が無いので、現在の選択を維持する。
  loadAudioIntoPlayer(next.url, { autoplay: false });
  const playedNode = deferredNode || findMessageByAudioUrl(next.url);
  if (playedNode) highlightSelectedMessage(playedNode);
  player.play().catch(() => {
    setStageStatus("tap play", playbackSlot);
    setSpeakingState(false);
  });
  if (autoMode && !autoPending && !hasBufferedOtherSlot(playbackSlot)) {
    window.setTimeout(() => continueAutoConversation(), 0);
  }
}

player.addEventListener("ended", playNext);
// 「…」＞再生速度で速度を変えたら記憶し、以後のすべての再生に適用する。
player.addEventListener("ratechange", () => {
  if (player.playbackRate) preferredPlaybackRate = player.playbackRate;
});
player.addEventListener("loadeddata", applyPreferredPlaybackRate);
document.addEventListener("pointerdown", unlockAudioPlayback, { passive: true, once: true });
document.addEventListener("keydown", unlockAudioPlayback, { passive: true, once: true });
document.addEventListener("touchstart", unlockAudioPlayback, { passive: true, once: true });

function slotForExternalEvent(event) {
  if (event.speakerSlot === "second") {
    if (!twoPlayerMode.checked) {
      twoPlayerMode.checked = true;
      updateTwoPlayerMode();
    }
    return "second";
  }
  return "main";
}

function handleExternalSpeakEvent(event) {
  const audios = Array.isArray(event.audios) ? event.audios : [];
  if (!audios.length) return;
  const slot = slotForExternalEvent(event);
  const timing = audios.map((item) => `${item.elapsed}s`).join(", ");
  const style = event.emojiStyle ? ` / style ${event.emojiStyle}` : "";
  const sourceSpeaker = event.speaker ? ` from ${event.speaker}` : "";
  const meta = `external speak${sourceSpeaker}${style} / pose ${event.expression || "neutral"} / tts ${timing}`;
  const primaryAudioUrl =
    (event.combined && event.combined.url) || (audios[0] && audios[0].url) || "";
  addMessage("assistant", event.text || audios.map((item) => item.text).join(""), meta, {
    audioUrl: primaryAudioUrl,
  });
  // 結合済み音声があれば、再生ボタン（▶）や保存ボタンがその1ファイルを対象にできるよう
  // 分割音声の代わりに結合ファイル1つだけをキューへ流す。
  const combined = event.combined && event.combined.url ? event.combined : null;
  const playItems = combined
    ? [
        {
          ...combined,
          text: combined.text || event.text || audios.map((item) => item.text).join(""),
          expression: combined.expression || event.expression || "neutral",
        },
      ]
    : audios;
  playQueue(playItems, slot, { append: true });
}

async function primeExternalSpeakEvents() {
  try {
    const res = await fetch("/api/speak-events?after=latest");
    const data = await res.json();
    if (res.ok) {
      externalSpeakLastId = Number(data.latestId || 0);
    }
  } catch {
    externalSpeakLastId = 0;
  }
}

async function pollExternalSpeakEvents() {
  if (externalSpeakPolling) return;
  externalSpeakPolling = true;
  try {
    const res = await fetch(`/api/speak-events?after=${externalSpeakLastId}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    const events = Array.isArray(data.events) ? data.events : [];
    for (const event of events) {
      externalSpeakLastId = Math.max(externalSpeakLastId, Number(event.id || 0));
      handleExternalSpeakEvent(event);
    }
    externalSpeakLastId = Math.max(externalSpeakLastId, Number(data.latestId || 0));
  } catch {
    // Keep this quiet; normal chat should not be interrupted by a polling miss.
  } finally {
    externalSpeakPolling = false;
  }
}

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const diagnostics = data.diagnostics || {};
    lmStatus.textContent = data.models.length
      ? `${data.models.length} models`
      : `not detected: ${data.lmStudioUrl}`;
    if (data.contextLimit) {
      // 環境デフォルトはキャラ別値のフォールバックとして保持。未編集時のみ欄へ反映。
      defaultContextLimit = Number(data.contextLimit) || defaultContextLimit;
      if (!contextLimit.dataset.touched) contextLimit.value = data.contextLimit;
    }
    updateContextUsage();
    if (data.ttsCaption && !ttsCaption.dataset.touched) {
      ttsCaption.value = data.ttsCaption;
    }
    mainReferencePath = mainReferencePath || data.reference || "";
    secondReferencePath = secondReferencePath || data.luviaReference || "";
    if (diagnostics.remoteLuviaEnabled && !secondTtsHost.dataset.touched) {
      ttsBackendMode.value = "remote";
      secondTtsHost.value = data.luviaRemoteTtsUrl || diagnostics.remoteLuviaUrl || secondTtsHost.value;
      updateTtsBackendControls();
    }
    renderCharacterEditor();
    if (data.irodoriReady) {
      const remoteLabel = diagnostics.remoteLuviaEnabled ? " / 2P remote on" : " / 2P local";
      irodoriStatus.textContent = `${data.referenceExists ? "refs ready" : "ref missing"} / ${data.irodoriRoot}${remoteLabel}`;
    } else {
      const missing = [];
      if (!diagnostics.gitExists) missing.push("git");
      if (!diagnostics.uvExists) missing.push("uv");
      if (!diagnostics.irodoriRootExists) missing.push("Irodori");
      if (diagnostics.irodoriRootExists && !diagnostics.irodoriPythonExists) missing.push("Irodori venv");
      irodoriStatus.textContent = `setup needed: ${missing.join(", ") || data.irodoriRoot}`;
    }
    expressionImages = Object.fromEntries(
      Object.entries(data.expressions || expressionImages).map(([key, value]) => [
        key,
        Array.isArray(value) ? value : [value],
      ])
    );
    setExpression("neutral");
    modelSelect.innerHTML = "";
    const codexOption = document.createElement("option");
    codexOption.value = "__codex_queue__";
    codexOption.textContent = "Codex (queue)";
    modelSelect.appendChild(codexOption);
    for (const model of data.models) {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      modelSelect.appendChild(option);
    }
    // サーバが環境変数（LM_STUDIO_MODEL）から解決した既定モデルを最優先で選ぶ。
    // 環境変数は GGUF のフルパスで指定されることがあり、/models が返す ID とは
    // 書き方が違うので、記号を落として「どちらかがどちらかを含む」で照合する。
    const normalizeModelId = (value) =>
      String(value || "")
        .toLowerCase()
        .replace(/\.gguf$/, "")
        .replace(/[^a-z0-9]/g, "");
    const preferredId = normalizeModelId(data.preferredModel);
    const preferredOption = !preferredId
      ? null
      : [...modelSelect.options]
          .filter((opt) => {
            const id = normalizeModelId(opt.value);
            return id.length >= 6 && (preferredId.includes(id) || id.includes(preferredId));
          })
          // QAT のファイル名は gemma-4-12b-it-QAT-Q4_0.gguf で it 側にも当たるため、
          // 両方ロードされている場合は限定の強い（長い）ID を採る。
          .sort((a, b) => normalizeModelId(b.value).length - normalizeModelId(a.value).length)[0];
    const preferred =
      [...modelSelect.options].find((opt) => opt.value === "__codex_queue__" && opt.dataset.preferred === "1") ||
      preferredOption ||
      [...modelSelect.options].find((opt) => opt.value === "gemma-4-12b-it") ||
      [...modelSelect.options].find((opt) => opt.value === "gemma-4-31b-it") ||
      [...modelSelect.options].find((opt) => opt.value.toLowerCase().includes("gemma"));
    if (preferred) modelSelect.value = preferred.value;

    emojiStyleSelect.innerHTML = '<option value="">plain</option>';
    for (const item of data.emojis || []) {
      const option = document.createElement("option");
      option.value = item.emoji;
      option.textContent = `${item.emoji} ${item.label}`;
      option.title = item.description;
      emojiStyleSelect.appendChild(option);
    }
  } catch (error) {
    lmStatus.textContent = "error";
    irodoriStatus.textContent = String(error);
  }
}

async function sendChatTurn({
  message,
  visibleUserText = message,
  slot = activeStage().slot,
  isAuto = false,
  allowWhileLocked = false,
  backgroundAuto = false,
  webSearchNow = false,
  webContext = "",
  webTopic = "",
}) {
  if (interactionLocked && !allowWhileLocked) return false;
  const text = String(message || "").trim();
  if (!text) return false;
  const historyBeforeSend = [...history];
  if (visibleUserText) {
    addMessage("user", visibleUserText);
    history.push({ role: "user", content: visibleUserText });
  }
  updateContextUsage();
  if (!backgroundAuto) {
    messageInput.value = "";
  }
  const stage = activeStage(slot);
  setActiveSpeaker(slot, isAuto || autoMode);
  if (!backgroundAuto) {
    setInteractionLocked(true, "Wait");
    setStageStatus("thinking", slot);
    setSpeakingState(false, slot);
    if (slot === "main") setExpression("soft");
  } else {
    sessionStatus.textContent = `auto thinking: ${stage.speaker}`;
  }

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history: historyBeforeSend,
        model: modelSelect.value,
        steps: stage.steps,
        speechRate: speechRate.value,
        replyLength: replyLength.value,
        llmGenerationMode: llmGenerationMode.value,
        speaker: stage.speaker,
        speakerSlot: stage.slot,
        characterId: stage.id,
        systemPrompt: stage.systemPrompt,
        userAddress: userAddress.value,
        ttsCaption: stage.ttsCaption,
        styleGuide: stage.styleGuide,
        cfgScaleText: stage.cfgScaleText,
        cfgScaleCaption: stage.cfgScaleCaption,
        cfgScaleSpeaker: stage.cfgScaleSpeaker,
        secondSystemPrompt: secondSystemPrompt.value,
        secondTtsCaption: secondTtsCaption.value,
        referencePath: stage.slot === "main" ? stage.referencePath : mainReferencePath,
        secondReferencePath: stage.slot === "second" ? stage.referencePath : secondReferencePath,
        twoPlayerMode: twoPlayerMode.checked,
        twoOnlyMode: twoPlayerMode.checked && twoOnlyMode.checked,
        ttsBackendMode: ttsBackendMode.value,
        secondTtsHost: secondTtsHost.value.trim(),
        contextLimit: Number(contextLimit.value || 8200),
        emojiStyle: autoEmoji.checked ? "" : currentEmojiStyle(),
        autoEmoji: autoEmoji.checked,
        webSearch: webSearchNow || (webSearch.checked && slot === "main" && !isAuto),
        webContext,
        webTopic,
        noDialogue: autoNoDialogue || wantsNoDialogue(text),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    lastContextStats = data.contextStats || lastContextStats;
    if (autoMode && data.webContext) {
      autoWebContext = data.webContext;
      autoWebQuery = data.webQuery || "";
      autoWebResults = data.webResults || [];
    }
    const timing = data.audios.map((item) => `${item.elapsed}s`).join(", ");
    const style = buildStyleMeta(data);
    const webQuery = String(data.webQuery || "");
    const webQueryLabel = webQuery.length > 32 ? `${webQuery.slice(0, 31)}…` : webQuery;
    const webMeta = data.webSearch || data.webContext
      ? ` / web ${data.webSearch ? (data.webResults || []).length : "memo"}${webQueryLabel ? ` q:${webQueryLabel}` : ""}`
      : "";
    const paceMeta = data.speechRate === "fast" ? " / pace fast" : "";
    const assistantMeta = `${data.speaker || mainCharacterName} / ${data.model} / ${data.replyLength}${style}${webMeta}${paceMeta} / pose ${data.expression} / tts ${timing}`;
    const annotatedReply = buildAnnotatedReply(data);
    const primaryAudioUrl =
      (data.combined && data.combined.url) ||
      (Array.isArray(data.audios) && data.audios[0] && data.audios[0].url) ||
      "";
    const regen = buildRegenPayload(data, stage);
    if (!backgroundAuto) {
      addMessage("assistant", annotatedReply, assistantMeta, { audioUrl: primaryAudioUrl, regen });
    }
    history.push({
      role: "assistant",
      content: `${data.speaker || stage.speaker}: ${data.reply}`,
      // リロード後も注釈・meta 行・再生対象・再生成パラメータを復元するための表示用メタ（LM context には非影響）。
      display: { text: annotatedReply, meta: assistantMeta, audioUrl: primaryAudioUrl, regen },
    });
    // 会話が1ターン追加されたので、Save ボタンを待たずに履歴を自動保存する。
    autoSaveSession();
    lastAssistantSpeaker = data.speaker || stage.speaker;
    lastAssistantText = data.reply;
    updateContextUsage();
    if (Array.isArray(data.audios) && data.audios.length) {
      // 結合済み音声があれば ▶ 再生・保存が1ファイルを対象にできるよう、結合ファイルだけを流す。
      const combined = data.combined && data.combined.url ? data.combined : null;
      let playItems = combined
        ? [
            {
              ...combined,
              text: combined.text || data.reply,
              expression: combined.expression || data.expression || "neutral",
            },
          ]
        : data.audios;
      if (backgroundAuto && playItems.length) {
        playItems = [...playItems];
        playItems[0] = {
          ...playItems[0],
          deferredMessage: { reply: annotatedReply, meta: assistantMeta, audioUrl: primaryAudioUrl, regen },
        };
      }
      playQueue(playItems, stage.slot, { append: backgroundAuto });
    } else {
      setStageStatus(data.codexQueued ? "codex queued" : "ready", stage.slot);
      setSpeakingState(false, stage.slot);
      if (!backgroundAuto && !interactionLocked) {
        setInteractionLocked(false);
      }
    }
    return true;
  } catch (error) {
    autoMode = false;
    autoPending = false;
    addMessage("assistant", `エラー: ${error.message}`);
    setStageStatus("error", stage.slot);
    if (!backgroundAuto) {
      setInteractionLocked(false);
    }
    updateAutoControls();
    return false;
  }
}

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;
  if (autoMode) {
    queueAutoTopic(text);
    return;
  }
  if (interactionLocked) return;
  await sendChatTurn({
    message: text,
    visibleUserText: text,
    slot: activeStage().slot,
    isAuto: false,
  });
});

async function startAutoConversation() {
  if (interactionLocked || autoMode || !twoPlayerMode.checked) return;
  const topic = messageInput.value.trim();
  if (!topic) {
    sessionStatus.textContent = "enter a topic first";
    return;
  }
  autoMode = true;
  autoPending = false;
  updateTwoPlayerMode();
  autoNextSlot = "main";
  lastAssistantSpeaker = "";
  lastAssistantText = "";
  autoTopic = topic;
  autoTopicQueue = [];
  autoWebContext = "";
  autoWebQuery = "";
  autoWebResults = [];
  autoTurnCount = 0;
  autoNoDialogue = wantsNoDialogue(topic);
  updateAutoControls();
  sessionStatus.textContent = "auto running";
  const firstSlot = autoNextSlot;
  autoNextSlot = otherSlot(firstSlot);
  autoTurnCount += 1;
  const firstAutoMessage = autoNoDialogue
    ? `${twoOnlyGuidance()}${noDialogueGuidance()}お題: ${autoTopic}\nこれは2人の自動進行の第${autoTurnCount}ターンです。通常の会話として返さず、発声・吐息・擬音の強弱、間、苦しさ、気持ちよさの変化だけで少し展開してください。`
    : `${twoOnlyGuidance()}お題: ${autoTopic}\nこれは2人の自動会話の第${autoTurnCount}ターンです。お題を会話の中心に置き、結論へ急がず、相手が次に返しやすい問い・感想・小さなズレを残して始めてください。`;
  await sendChatTurn({
    message: firstAutoMessage,
    visibleUserText: `お題: ${topic}`,
    slot: firstSlot,
    isAuto: true,
    webSearchNow: webSearch.checked,
    webTopic: topic,
  });
}

async function continueAutoConversation() {
  if (!autoMode || autoPending) return;
  autoPending = true;
  const slot = autoNextSlot;
  autoNextSlot = otherSlot(slot);
  autoTurnCount += 1;
  const queuedTopic = consumeQueuedAutoTopic();
  const shouldRefreshWeb = Boolean(queuedTopic && webSearch.checked);
  const speaker = characterNameForSlot(slot);
  const partner = characterNameForSlot(otherSlot(slot));
  sessionStatus.textContent = queuedTopic ? `auto: ${speaker} / new topic` : `auto: ${speaker}`;
  const previousLine = lastAssistantText
    ? `直前に${partner}がこう言いました。\n「${lastAssistantText}」\n`
    : "";
  const topicLine = queuedTopic
    ? `ここから新しいお題に切り替えます。新しいお題は「${autoTopic}」です。第${autoTurnCount}ターンとして、直前の発言を受けつつ、この新しいお題へ自然に寄せてください。\n`
    : autoTopic
      ? `会話のお題は「${autoTopic}」です。第${autoTurnCount}ターンとして、このお題から離れすぎず、直前の発言を受けて少しだけ展開を進めてください。\n`
      : "";
  const nextAutoMessage = autoNoDialogue
    ? `${twoOnlyGuidance()}${noDialogueGuidance()}${queuedTopic ? "ここから新しいお題に切り替えます。" : ""}会話のお題は「${autoTopic}」です。第${autoTurnCount}ターンです。直前の発声を受けて、普通のセリフではなく、発声・吐息・擬音の流れだけを少し変化させて続けてください。呼びかけ、質問、説明、選択肢提示は禁止です。`
    : `${twoOnlyGuidance()}${topicLine}${previousLine}あなたは${speaker}です。${partner}の発言を受けて、${partner}に返す一言として自然に会話を続けてください。単純な相槌で終わらせず、前の発言から一歩だけ発展させてください。新しい情報、疑問、軽い反論、感情の変化のどれかを少し入れて、次の発言につながる余韻を残してください。`;
  try {
    await sendChatTurn({
      message: nextAutoMessage,
      visibleUserText: "",
      slot,
      isAuto: true,
      allowWhileLocked: true,
      backgroundAuto: true,
      webSearchNow: shouldRefreshWeb,
      webContext: shouldRefreshWeb ? "" : autoWebContext,
      webTopic: shouldRefreshWeb ? queuedTopic : "",
    });
  } finally {
    autoPending = false;
    updateAutoControls();
  }
}

function stopAutoConversation() {
  autoMode = false;
  autoPending = false;
  autoTopic = "";
  autoTopicQueue = [];
  autoWebContext = "";
  autoWebQuery = "";
  autoWebResults = [];
  autoTurnCount = 0;
  autoNoDialogue = false;
  updateAutoControls();
  sessionStatus.textContent = interactionLocked ? "auto stopping" : "auto stopped";
}

autoEmoji.addEventListener("change", refreshEmojiInputs);
twoPlayerMode.addEventListener("change", updateTwoPlayerMode);
ttsBackendMode.addEventListener("change", updateTtsBackendControls);
secondTtsHost.addEventListener("input", () => {
  secondTtsHost.dataset.touched = "1";
  updateTtsBackendControls();
});
openOptionsButton.addEventListener("click", openOptions);
closeOptionsButton.addEventListener("click", closeOptions);
optionsModal.addEventListener("click", (event) => {
  if (event.target === optionsModal) {
    closeOptions();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !optionsModal.hidden) {
    closeOptions();
  }
});
mainCharacterSelect.addEventListener("change", () => {
  const previousId = activeMainCharacterId;
  activeMainCharacterId = mainCharacterSelect.value;
  syncActiveCharacterState();
  if (previousId !== activeMainCharacterId) {
    // メインキャラを変えたら、直前キャラのログを退避して新キャラのログへ切り替える。
    switchCharacterHistory(previousId, activeMainCharacterId);
  }
});
secondCharacterSelect.addEventListener("change", () => {
  activeSecondCharacterId = secondCharacterSelect.value;
  syncActiveCharacterState();
});
editCharacterSelect.addEventListener("change", () => {
  editingCharacterId = editCharacterSelect.value;
  renderCharacterEditor();
});
newCharacterButton.addEventListener("click", makeNewCharacter);
loadCharactersButton.addEventListener("click", async () => {
  try {
    await loadCharacters();
    sessionStatus.textContent = `loaded ${characterList().length} characters`;
  } catch (error) {
    sessionStatus.textContent = `character load error: ${error.message}`;
  }
});
saveCharactersButton.addEventListener("click", async () => {
  try {
    await saveCharacters();
  } catch (error) {
    sessionStatus.textContent = `character save error: ${error.message}`;
  }
});
for (const input of [
  editCharacterName,
  editSystemPrompt,
  editTtsCaption,
  editStyleGuide,
  editSteps,
  editCfgScaleText,
  editCfgScaleCaption,
  editCfgScaleSpeaker,
]) {
  input.addEventListener("input", updateEditingCharacter);
}
for (const button of speechRateButtons) {
  button.addEventListener("click", () => setSpeechRate(button.dataset.rate));
}
editExpressionSelect.addEventListener("change", renderExpressionThumbs);
addExpressionSlot.addEventListener("click", addExpressionSlotForEditingCharacter);
installReferenceDrop(editReferenceDrop, editReferenceFile, editReferenceChoose, "edit");
installImageDrop(expressionImageDrop, expressionImageFile, expressionImageChoose);
shutdownAppButton.addEventListener("click", async () => {
  const ok = window.confirm("IrodoriTTS UI とこのチャットアプリを終了します。よろしいですか？");
  if (!ok) return;
  shutdownAppButton.disabled = true;
  shutdownAppButton.textContent = "Stopping...";
  sessionStatus.textContent = "shutting down";
  try {
    await fetch("/api/shutdown", { method: "POST" });
    document.body.classList.add("is-shutdown");
    sessionStatus.textContent = "stopped";
  } catch (error) {
    shutdownAppButton.disabled = false;
    shutdownAppButton.textContent = "終了";
    sessionStatus.textContent = `shutdown error: ${error.message}`;
  }
});
autoStartButton.addEventListener("click", () => {
  if (autoMode) {
    stopAutoConversation();
  } else {
    startAutoConversation();
  }
});
contextLimit.addEventListener("input", () => {
  contextLimit.dataset.touched = "1";
  // 編集値を即アクティブキャラの値として保持（切替/保存前に失わないように）。
  characterContextLimits[activeMainCharacterId] = currentContextLimitValue();
  updateContextUsage();
});

ttsCaption.addEventListener("input", () => {
  ttsCaption.dataset.touched = "1";
});

secondTtsCaption.addEventListener("input", () => {
  secondTtsCaption.dataset.touched = "1";
});

saveSessionButton.addEventListener("click", async () => {
  try {
    sessionStatus.textContent = "saving...";
    await saveSession();
  } catch (error) {
    sessionStatus.textContent = `save error: ${error.message}`;
  }
});

loadSessionButton.addEventListener("click", async () => {
  try {
    sessionStatus.textContent = "loading...";
    // 手動 Load では選択中のキャラを維持し、そのキャラのログを表示する。
    await loadSession(false, true);
  } catch (error) {
    sessionStatus.textContent = `load error: ${error.message}`;
  }
});

clearContextButton.addEventListener("click", clearContext);

saveAudioButton.addEventListener("click", async () => {
  const currentUrl = player.getAttribute("src") || "";
  if (!currentUrl) {
    audioSaveStatus.textContent = "no audio";
    return;
  }
  try {
    audioSaveStatus.textContent = "saving...";
    const res = await fetch("/api/save-audio", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: currentUrl,
        label: twoPlayerMode.checked ? "2p" : "rinon",
        // 保存音声をキャラクター別フォルダへ振り分けるためメインキャラ ID を送る。
        characterId: activeMainCharacterId,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    audioSaveStatus.textContent = `saved: ${data.name}`;
  } catch (error) {
    audioSaveStatus.textContent = `save error: ${error.message}`;
  }
});

// 選択中の返答（node）に対応する history エントリを、表示用 audioUrl の一致で探す。
// 生成ファイル名はタイムスタンプ付きで一意なため、URL 一致で確実に対応付けられる。
function findHistoryEntryByAudioUrl(url) {
  if (!url) return null;
  return (
    history.find((entry) => entry && entry.display && entry.display.audioUrl === url) || null
  );
}

// 生成時パラメータ（_regen）が無い過去の返答向けに、履歴テキストと現在のキャラ設定から
// 再生成データを組み立てる。感情セグメントは復元できないため空（＝返答全体を1発話で合成）とし、
// 読み・表現のみを作り直す。話者は履歴の "話者名: 本文" 先頭から推定し、そのキャラ設定を使う。
function buildFallbackRegen(node) {
  const url = node && node.dataset ? node.dataset.audioUrl || "" : "";
  const entry = findHistoryEntryByAudioUrl(url);
  let reply = "";
  let speakerName = "";
  if (entry && entry.content) {
    const match = entry.content.match(/^([^:：]+)[:：]\s*([\s\S]*)$/);
    if (match) {
      speakerName = match[1].trim();
      reply = match[2].trim();
    } else {
      reply = entry.content.trim();
    }
  }
  if (!reply) return null;
  const stage = activeStage(slotForSpeakerName(speakerName));
  return {
    reply,
    segments: [],
    emojiStyle: "",
    llmEmojiStyle: "",
    speaker: stage.speaker,
    speakerSlot: stage.slot,
    ttsCaption: stage.ttsCaption,
    cfgScaleText: stage.cfgScaleText,
    cfgScaleCaption: stage.cfgScaleCaption,
    cfgScaleSpeaker: stage.cfgScaleSpeaker,
    steps: stage.steps,
    referencePath: stage.slot === "main" ? stage.referencePath : mainReferencePath,
    secondReferencePath: stage.slot === "second" ? stage.referencePath : secondReferencePath,
    speechRate: "normal",
  };
}

// irodori 再生成：選択中の返答テキストを LLM を介さず TTS のみで作り直し、音声を差し替える。
// 辞書追加やコード修正の後に、同じ返答へ最新の読み・表現を反映させたいときに使う。
regenAudioButton.addEventListener("click", async () => {
  const node = selectedMessageNode;
  let regen = node && node._regen;
  if (node && !regen) {
    // 過去の会話（再生成パラメータ未保存）はフォールバックで組み立て、次回以降のため枠に保持する。
    regen = buildFallbackRegen(node);
    if (regen) node._regen = regen;
  }
  if (!node || !regen) {
    audioSaveStatus.textContent = "no target";
    return;
  }
  if (regenBusy) return;
  regenBusy = true;
  updateSelectionActions();
  const oldUrl = node.dataset.audioUrl || "";
  audioSaveStatus.textContent = "regenerating...";
  try {
    const res = await fetch("/api/regenerate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...regen,
        // リモート TTS（2P）設定は再生成時点の UI 値を使う。
        ttsBackendMode: ttsBackendMode.value,
        secondTtsHost: secondTtsHost.value.trim(),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    const newUrl =
      (data.combined && data.combined.url) ||
      (Array.isArray(data.audios) && data.audios[0] && data.audios[0].url) ||
      "";
    if (!newUrl) throw new Error("no audio returned");
    node.dataset.audioUrl = newUrl;
    if (Array.isArray(data.segments)) node._regen = { ...regen, segments: data.segments };
    // history 側の表示用 URL と再生成パラメータも更新し、保存/リロード後も差し替えを保持する。
    const entry = findHistoryEntryByAudioUrl(oldUrl);
    if (entry && entry.display) {
      entry.display.audioUrl = newUrl;
      if (entry.display.regen) entry.display.regen = node._regen;
    }
    // 選択中だった場合はプレイヤーを差し替え、再生成結果をそのまま鳴らす。
    if (selectedAudioUrl === oldUrl || selectedMessageNode === node) {
      loadAudioIntoPlayer(newUrl, { autoplay: true });
    }
    audioSaveStatus.textContent = "regenerated";
  } catch (error) {
    audioSaveStatus.textContent = `regen error: ${error.message}`;
  } finally {
    regenBusy = false;
    updateSelectionActions();
  }
});

// 削除：選択中の返答を、対応するユーザー発言（お題）ごと会話から取り除く。
// 誤操作防止のため (y/N) 確認を必須にし、ENTER（空入力）や y 以外は削除しない。
//
// 返答だけを消すとユーザー発言が宛先を失って履歴に残り、次の想起や再構築で
// 「問いかけだけがある往復」として扱われてしまう。RAG の記憶は往復単位で保存する
// ため、削除も往復単位で揃える。
// ただし2人だけモードは「お題1つに複数キャラの返答が続く」ので、他の返答がまだ
// 残る場合はお題を消さない（残った返答が宛先を失うため）。
deleteMessageButton.addEventListener("click", () => {
  const node = selectedMessageNode;
  if (!node) {
    audioSaveStatus.textContent = "no target";
    return;
  }
  const url = node.dataset.audioUrl || "";
  const entry = findHistoryEntryByAudioUrl(url);
  const idx = entry ? history.indexOf(entry) : -1;

  // この返答に紐づくユーザー発言（直前の user）と、それを共有する他の返答を調べる。
  let userIdx = -1;
  let siblingReplies = 0;
  if (idx >= 0) {
    for (let i = idx - 1; i >= 0; i -= 1) {
      if (history[i] && history[i].role === "user") {
        userIdx = i;
        break;
      }
    }
    if (userIdx >= 0) {
      for (let i = userIdx + 1; i < history.length; i += 1) {
        if (!history[i] || history[i].role !== "assistant") break;
        if (i !== idx) siblingReplies += 1;
      }
    }
  }
  const removeUser = userIdx >= 0 && siblingReplies === 0;

  const answer = window.prompt(
    removeUser
      ? "この往復（あなたの発言と返答）を削除しますか？ (y/N)"
      : "この返答を削除しますか？（お題は他の返答が使うので残します） (y/N)",
    ""
  );
  if (answer === null || answer.trim().toLowerCase() !== "y") {
    audioSaveStatus.textContent = "delete canceled";
    return;
  }

  // DOM から対応するユーザー発言の枠も消す。history とDOMの並びは一致しない場合が
  // あるため（自動会話のバックグラウンド返答は枠を作らず履歴にだけ積まれる）、
  // インデックスではなく直前の .msg.user を辿って特定する。
  if (removeUser) {
    let prev = node.previousElementSibling;
    while (prev && !prev.classList.contains("user")) {
      prev = prev.previousElementSibling;
    }
    if (prev) prev.remove();
  }
  node.remove();

  // 大本のログを照合するための本文を、履歴から取り除く前に控えておく。
  // chat.jsonl は「1行 = ユーザー発言＋返答」なので、返答だけを消す場合でも
  // 対応する行を特定するにはユーザー発言が必要（お題は他の行にも現れる）。
  const removedReplyText = entry ? String(entry.content || "") : "";
  const removedUserText =
    userIdx >= 0 && history[userIdx] ? String(history[userIdx].content || "") : "";

  // 履歴からも削除する。ユーザー発言を先に消すと返答の位置がずれるので、
  // 後ろ（返答）から順に取り除く。
  if (idx >= 0) {
    history.splice(idx, 1);
    if (removeUser) history.splice(userIdx, 1);
  }

  // 削除した枠が再生対象だった場合はプレイヤーを空にする。
  if (selectedAudioUrl === url) {
    player.pause();
    player.removeAttribute("src");
    player.load();
    selectedAudioUrl = "";
  }
  selectedMessageNode = null;
  updateSelectionActions();
  updateContextUsage();
  // 削除を history.json へ即時反映する（保存を待つ間に別の同期や再構築が走ると、
  // 消したはずの会話が復活する）。
  autoSaveSession();
  // 大本の生ログ・感情ログ・RAG検索DB からも取り除く。ここを省くと chat.jsonl に
  // 残り続け、後日そこから再構築したときに削除した会話が復活する。
  // 自動保存とは別経路で「削除した」ことだけを伝える（自動保存は現在の会話
  // コンテキストを書くだけなので、そこから削除を推論すると Clear Context と
  // 区別できず、残すべき生ログまで消えてしまう）。
  deleteTurnRecords({
    userText: removedUserText,
    replyText: removedReplyText,
    audioUrl: url,
  });
  audioSaveStatus.textContent = removeUser ? "deleted (turn)" : "deleted (reply)";
});

// 削除した往復を大本のログと検索DBからも消すようサーバへ依頼する。
// 失敗しても画面と history.json の削除は成立しているので、会話は止めずに警告だけ出す
// （復旧したいときは tools/sync_memory.py --source history --propagate で追いつける）。
async function deleteTurnRecords({ userText, replyText, audioUrl }) {
  if (!userText && !audioUrl) return;
  try {
    const res = await fetch("/api/delete-turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        characterId: activeMainCharacterId,
        speaker: mainCharacterName,
        twoOnlyMode: twoPlayerMode.checked && twoOnlyMode.checked,
        userText: userText || "",
        replyText: replyText || "",
        audioUrl: audioUrl || "",
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    const detail = `log ${data.chatLog} / mem ${data.memories} / fact ${data.facts}`;
    audioSaveStatus.textContent = `deleted (${detail})`;
  } catch (error) {
    audioSaveStatus.textContent = `delete sync failed: ${error.message}`;
  }
}

messageInput.addEventListener("compositionstart", () => {
  messageInputComposing = true;
});

messageInput.addEventListener("compositionend", () => {
  messageInputComposing = false;
});

messageInput.addEventListener("keydown", (event) => {
  if (eventMatchesSendShortcut(event)) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

async function initialize() {
  refreshEmojiInputs();
  await loadCharacters();
  updateTwoPlayerMode();
  await refreshStatus();
  await loadSession(true);
  updateTtsBackendControls();
  await primeExternalSpeakEvents();
  window.setInterval(pollExternalSpeakEvents, 1500);
}

initialize().catch((error) => {
  sessionStatus.textContent = `init error: ${error.message}`;
});
