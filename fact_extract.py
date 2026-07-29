"""往復から「誰が・誰に・何を・どうした」を取り出す主客ハイブリッド抽出。

この会話ログで最も間違えやすいのは事実そのものではなく**主客**である。
「俺が君に作ってあげた料理」と「君が俺に作ってくれた料理」は、意味検索では
ほぼ同じベクトルになるため、cosine 類似度では区別できない。その結果、
キャラクターが「私は料理を作っていません」と主客を取り違えて答えたり、
ユーザーが作った料理を自分の手柄として語ったりする。

そこで抽出の段で向き（direction）を確定させ、事実台帳（rag_memory.facts）へ
構造として保存する。想起は SQL の絞り込みになるので、主客が原理的にブレない。

抽出は 2 段のハイブリッド:
  1) ルール（本モジュールの正規表現）
     日本語の授受表現は主客の**極めて強い信号**である。「〜してあげた」は
     発話者→相手、「〜してくれた／もらった」は相手→発話者。これを発話の role
     （user / assistant）と組み合わせると向きが一意に決まる。
       ・user 発話の「作ってあげた」      → user->char
       ・user 発話の「作ってくれた」      → char->user
       ・assistant 発話の「作ってあげた」 → char->user
       ・assistant 発話の「作ってくれた」 → user->char
     LLM 呼び出しゼロ・数ミリ秒で、授受表現がある往復はここで確定する。
  2) LLM（ルールで決まらなかった往復だけ）
     授受表現が省略された文（「今日は肉じゃが作ったよ」）は、ルールでは
     向きも客体も確定できない。この分だけをローカル LLM に回して JSON で取る。
     全往復を LLM に流すより桁で速く、精度は落とさない。

本モジュールは HTTP を持たない。LLM 呼び出しは ``llm`` 引数（callable）として
呼び出し側（app.py / tools/build_fact_ledger.py）から注入する。依存方向を
一方向に保ち、app.py 抜きでもツールから同じ抽出を再現できるようにするため。

抽出できなかった要素は捨てず、``''`` / ``direction='unknown'`` のまま残す。
台帳から消すと「台帳に無い＝存在しない」という新しい取りこぼしを作るためで、
不明は不明として保存し、提示のときに「主客不明」と明示する。
"""

from __future__ import annotations

import json
import re

# --- 語彙定義（拡張はここだけを触れば済むようにまとめる）----------------------
# 行為の動詞。キーが台帳へ入る正規化後の verb、値は表層形のパターン。
# 語尾は活用を吸収するため「語幹＋任意の送り」で書く。
_VERB_PATTERNS: tuple[tuple[str, str], ...] = (
    ("作る", r"作(?:っ|り|る|れ)|つく(?:っ|り|る)|焼(?:い|く|け)|煮(?:た|て|込)|炒め|揚げ|蒸し|漬け"),
    ("買う", r"買(?:っ|う|い)|購入|買ってき"),
    ("渡す", r"渡(?:し|す)|贈(?:っ|る)|プレゼント|届け"),
    ("食べる", r"食べ|いただ(?:い|く)|食し"),
    ("行く", r"行(?:っ|く)|出かけ|訪れ|連れ(?:て|去)"),
    # 出かけた先での体験は「いつ・どこで何をしたか」の中心なので拾う。
    # ここに動詞が無いと質問側でも verb を検出できず、台帳の絞り込みが効かない
    # （実測: 「マリンタワーに登ったのはいつ？」で絞り込めず無関係な事実が並んだ）。
    ("登る", r"登(?:っ|る|り)|上が(?:っ|る)"),
    ("泊まる", r"泊(?:まっ|まる|めた)|宿泊"),
    ("撮る", r"撮(?:っ|る|り)"),
    ("会う", r"会(?:っ|う)|出会(?:っ|う)|待ち合わせ"),
    ("約束する", r"約束"),
    # 「言う」「見る」はルール抽出の対象にしない。会話ログそのものが「言ったこと」の
    # 記録なので事実として冗長なうえ、実データではゴミ客体の最大の発生源だった
    # （言う:こうして私 / 言う:おやすか / 見る:渡された飲み物 等）。
    # 約束・取り決めのように後から参照する価値がある発話は、LLM 抽出側が拾う。
    ("歌う", r"歌(?:っ|う)"),
    ("直す", r"直(?:し|す)|修理|治し"),
    ("壊す", r"壊(?:し|す)|破壊|割(?:っ|る)"),
    ("貸す", r"貸(?:し|す)"),
    ("送る", r"送(?:っ|る)"),
    ("読む", r"読(?:ん|む|み)"),
    ("書く", r"書(?:い|く)"),
)
# 客体の語からカテゴリを粗く推定する辞書。ここに無い語はカテゴリ '' のまま入れる
# （カテゴリは絞り込みの補助であって、無くても object/verb/direction で引ける）。
_CATEGORY_HINTS: tuple[tuple[str, str], ...] = (
    ("本", r"本|小説|漫画|マンガ|雑誌|写真集|図鑑|絵本|文庫|新書|詩集"),
    # 衣類は料理より前に置く。日本語には語境界が無いため「パンツ」が料理側の「パン」に
    # 部分一致して [料理] と判定される（実測）。先にここで拾って打ち切る。
    (
        "衣類",
        r"パンツ|下着|パンティ|ブラ|服|シャツ|ズボン|スカート|ドレス|コート|上着|"
        r"靴下|靴|帽子|水着|手袋|マフラー|エプロン|制服|着替え",
    ),
    (
        "料理",
        # 「パン」は「パンツ」「パンティ」に誤爆するので否定先読みで守る。
        r"料理|ご飯|ごはん|飯|おかず|晩御飯|晩ごはん|夕飯|朝食|昼食|弁当|"
        r"うどん|そば|ラーメン|カレー|シチュー|パスタ|スパゲ|餃子|チャーハン|"
        r"味噌汁|スープ|サラダ|煮物|炒め|天ぷら|唐揚げ|卵焼き|オムレツ|ハンバーグ|"
        r"肉じゃが|麻婆|丼|寿司|刺身|焼き魚|塩じゃけ|鮭|さけ|たたき|鍋|おでん|"
        r"ケーキ|クッキー|プリン|パン(?![ツち])|ピザ|グラタン|コロッケ|春巻|パエリア",
    ),
    ("飲み物", r"コーヒー|紅茶|お茶|ジュース|ココア|ミルク|ビール|ワイン|水|スムージー"),
    ("贈り物", r"プレゼント|贈り物|花束|指輪|ネックレス|ぬいぐるみ|人形|服|靴|鞄|かばん"),
    ("薬", r"薬|錠剤|サプリ|漢方"),
    ("音楽", r"歌|曲|音楽|ピアノ|ギター|アルバム|ライブ"),
    ("場所", r"公園|海|山|川|映画館|水族館|動物園|遊園地|神社|寺|喫茶店|カフェ|レストラン|旅行"),
)
# 授受表現（方向の決定に使う）。発話者から相手への行為か、その逆か。
# 「〜てあげる／やる」は発話者→相手、「〜てくれる／もらう／いただく」は相手→発話者。
_GIVE_RE = re.compile(r"(?:て|で)(?:あげ|やっ|やる|あげる)|してあげ|プレゼントし")
_RECEIVE_RE = re.compile(r"(?:て|で)(?:くれ|もらっ|もらう|いただ)|してくれ|もらった")
# 場所へ向かう／場所で行う動詞。これらの客体は「を」ではなく「に」「へ」を伴う
# （「マリンタワーに登った」「展望室へ行った」）ので、客体の抽出で助詞を広げる。
_LOCATIVE_VERBS = {"行く", "登る", "泊まる", "会う"}
# 客体と動詞の間に別の動詞が挟まっていないかを見るための全動詞パターン。
_ANY_VERB_RE = re.compile("|".join(pattern for _verb, pattern in _VERB_PATTERNS))
# 客体（何を）の抽出。助詞「を」の直前の名詞句を拾う素朴な規則。
# 20 文字までに制限し、句読点・助詞境界で切る（文全体を客体にしない）。
_OBJECT_RE = re.compile(r"([^、。！？\n\s「」『』]{1,20}?)を(?=[^、。]{0,12}?(?:%s))" % "|".join(
    pattern for _verb, pattern in _VERB_PATTERNS
))
# 客体から落とす前置きの助詞・修飾（「今日は肉じゃがを」→「肉じゃが」）。
# 「の」は対象にしない: 連体修飾として名詞句の一部になることが多く、落とすと
# 「星の王子さま」が「王子さま」に削れて書名が壊れる。
_OBJECT_TRIM_RE = re.compile(r"^.*?(?:は|も|が|に|で|と|、)(?=[^はもがにでと]{1,20}$)")
# 客体の頭に付いた連体修飾を落とす（「焼いたクッキー」→「クッキー」）。
_OBJECT_MODIFIER_RE = re.compile(r"^[^、。]*?(?:った|って|いた|えた|べた|した|める|ける)(?=.{2,12}$)")
# 授受表現の直後に客体が来る形（「作ってくれたオムライス」→「オムライス」）。
# 「を」を伴わないため _OBJECT_RE では取れない語をここで拾う。
# 対象は漢字・カタカナ語だけに限る: ひらがなを許すと助詞との境界が判定できず
# （「肉じゃが」の「が」と主格の「が」は区別できない）、「オムライスが美味しかった」を
# まるごと客体にしてしまう。ひらがな混じりの名詞は「を」経路（_OBJECT_RE）側で拾う。
_OBJECT_AFTER_RE = re.compile(r"(?:くれた|あげた|もらった|くれて|あげて|もらって)([一-鿿ァ-ヴー]{2,12})")
# 主体が明示されている場合の手掛かり（「〇〇が作った」）。
_SUBJECT_RE = re.compile(r"([一-鿿ぁ-んァ-ヴー]{2,12})が[^、。]{0,12}?(?:%s)" % "|".join(
    pattern for _verb, pattern in _VERB_PATTERNS
))
# 主体候補の頭に付きやすい時間・頻度の副詞（「あのときナデシコ」→「ナデシコ」）。
_TIME_PREFIX_RE = re.compile(
    r"^(?:あのとき|あの時|あの日|この前|こないだ|以前|昨日|今日|明日|昔|さっき|"
    r"先日|去年|今年|最近|いつも|たまに|よく|前に|まえに)+"
)
# 1 文字でも客体として意味を持つのは漢字・カタカナ（薬・本・パン等）。ひらがな 1 文字は
# 助詞や語尾の切れ端にすぎないので落とす。
_KANJI_RE = re.compile(r"[一-鿿々-〇]")
_KATAKANA_RE = re.compile(r"[ァ-ヴー]")
# 客体として意味を持たない語。助詞・形式名詞・代名詞は、素朴な「を」の直前を取る規則の
# 副作用で拾ってしまう（実データで「の」「より」「気」「まま」等が大量に混ざった）。
# 台帳に入れても「何があったか」を思い出す手がかりにならず、列挙の邪魔になるだけなので捨てる。
_OBJECT_STOPWORDS = {
    "の", "こと", "もの", "とき", "時", "ため", "よう", "まま", "ほう", "方", "うち",
    "ところ", "より", "ほど", "くらい", "ぐらい", "だけ", "など", "気", "中", "上", "下",
    "前", "後", "隣", "そば", "側", "感じ", "つもり", "はず", "わけ", "せい", "おかげ",
    "私", "わたし", "僕", "ぼく", "俺", "おれ", "あなた", "君", "きみ", "自分", "相手",
    "二人", "ふたり", "みんな", "誰か", "何か",
    "何", "なに", "誰", "だれ", "どこ", "いつ", "それ", "これ", "あれ", "そう", "こう",
    # 行為はしたが「何があったか」を示さない対象。LLM 抽出が返しがちなので落とす
    # （実データで「言う: 挨拶」「思う: 気持ち」が量産された）。
    "話", "言葉", "声", "挨拶", "あいさつ", "返事", "気持ち", "思い", "様子", "笑顔",
    "顔", "目", "手", "頭", "体", "心", "お礼", "感謝", "心配", "質問", "答え", "名前",
    "会話", "やり取り", "冗談", "説明", "報告", "反応", "態度", "表情",
}
# 抽出の切り出しに失敗して先頭に助詞の残骸が付くことがある
# （「〜から俺の顔を見た」→「ら俺の顔」、「〜のの中止」→「の中止」）。
# 剥がすのは「1 文字の助詞のあとに漢字・カタカナが続く」形だけに限る。
# ひらがなが続く場合まで剥がすと「のりまき」→「りまき」「らしさ」→「しさ」のように
# 正当な語を壊すため、そこは取りこぼしを受け入れる。
_OBJECT_HEAD_PARTICLE_RE = re.compile(r"^[らかがをにはもとでへやの](?=[^ぁ-んァ-ヴー])")

# 客体が総称語のままだと列挙に使えない（「本」ではどの本か分からない）。
# ルールで向きが取れても、客体がこれらだけなら LLM 抽出へ回して具体名を取りに行く。
_VAGUE_OBJECTS = {
    "本",
    "料理",
    "ご飯",
    "ごはん",
    "飯",
    "おかず",
    "食事",
    "もの",
    "物",
    "それ",
    "これ",
    "あれ",
    "プレゼント",
    "贈り物",
}
# 主体・客体として使えない総称語（これらは名前として台帳へ入れない）。
_GENERIC_NAMES = {
    "",
    "あなた",
    "きみ",
    "君",
    "お前",
    "おまえ",
    "わたし",
    "私",
    "僕",
    "ぼく",
    "俺",
    "おれ",
    "自分",
    "相手",
    "二人",
    "ふたり",
}


def _clean_object(text: str) -> str:
    """抽出した客体を名詞句へ寄せ、客体として使えない語なら空文字を返す。

    前置きの助詞・連体修飾を落としたうえで、助詞・形式名詞・代名詞（_OBJECT_STOPWORDS）
    と、ひらがな 1 文字だけの語を捨てる。ここを通さないと「の」「より」「気」のような
    無意味な行が台帳に溜まり、列挙質問の邪魔になる。
    """
    body = str(text or "").strip()
    body = _OBJECT_TRIM_RE.sub("", body).strip()
    body = _OBJECT_MODIFIER_RE.sub("", body).strip()
    body = body.strip("、。・…「」『』\"'`（）() 　")
    # 「のや観た景色」のように助詞の残骸が 2 つ重なることがあるので繰り返し剥がす
    # （剥がし過ぎを防ぐため 2 回まで）。
    for _ in range(2):
        trimmed = _OBJECT_HEAD_PARTICLE_RE.sub("", body).strip()
        if trimmed == body:
            break
        body = trimmed
    if not body or body in _OBJECT_STOPWORDS:
        return ""
    # ひらがな 1 文字は助詞・語尾の切れ端でしかない（漢字 1 字は「薬」「本」等で有効）。
    if len(body) == 1 and not _KANJI_RE.match(body) and not _KATAKANA_RE.match(body):
        return ""
    return body


def _guess_category(object_text: str, verb: str) -> str:
    """客体の語からカテゴリを粗く推定する（当たらなければ空文字）。"""
    body = str(object_text or "")
    for category, pattern in _CATEGORY_HINTS:
        if re.search(pattern, body):
            return category
    # 客体からは分からないが、動詞から言えることだけ補う。
    if verb == "作る" and body:
        return "料理"  # この会話ログで「作る」の客体は大半が食事
    return ""


def _find_verbs(text: str) -> list[str]:
    """本文に現れる行為の動詞（正規化後）を出現順で返す（質問側の絞り込み推定に使う）。"""
    found: list[tuple[int, str]] = []
    for verb, pattern in _VERB_PATTERNS:
        match = re.search(pattern, text)
        if match:
            found.append((match.start(), verb))
    found.sort()
    return [verb for _pos, verb in found]


def _find_verb_object_pairs(text: str) -> list[tuple[str, str, str]]:
    """「客体＋を＋動詞」の係り受けを保ったまま (動詞, 客体, 授受のヒント) の組を返す。

    動詞と客体をそれぞれ独立に集めて総当たりで組み合わせてはいけない。1 つの往復に
    複数の動詞があるだけで誤った事実が量産される（実測: 「パンツ」1 語に対して
    食べる/作る/買う/行く の 4 件が生成された）。客体は、その動詞に係っているものだけを取る。

    授受のヒント（'give' / 'receive' / ''）も動詞ごとに返す。文全体で授受表現を探すと、
    別の行為に係る向きを持ち込んでしまう（実測: 「何でも喜んで食べて**くれる**から、
    …パンツを買いに行ってくるね」で、買う行為が「ルリが買った」になった）。
    向きの手掛かりは、その動詞の直後にある授受表現だけに限る。
    """
    pairs: list[tuple[str, str, str]] = []
    for verb, pattern in _VERB_PATTERNS:
        # 場所へ向かう動詞は「〜に登る」「〜へ行く」の形を取るので助詞を広げる。
        particle = "[をにへ]" if verb in _LOCATIVE_VERBS else "を"
        # 「〜を（間に最大12文字）＋その動詞」。句読点は越えない。
        for match in re.finditer(
            r"([^、。！？\n\s「」『』]{1,20}?)"
            + particle
            + r"([^、。！？\n]{0,12}?)(?:"
            + pattern
            + ")",
            text,
        ):
            # 客体と動詞の間に別の動詞があれば、客体はそちらに係っている。
            # 「パンツを買いに行った」を「行く: パンツ」にしないため
            # （実測で「パンツ」に 食べる/作る/買う/行く の 4 件が付いた名残）。
            if _ANY_VERB_RE.search(match.group(2)):
                continue
            obj = _clean_object(match.group(1))
            if obj:
                pairs.append((verb, obj, _giving_hint(text, match.end())))
        # 「作ってくれたオムライス」のように、その動詞＋授受表現の直後に客体が来る形。
        for match in re.finditer(
            r"(?:" + pattern + r")[^、。！？\n]{0,4}?"
            r"(くれた|あげた|もらった|くれて|あげて|もらって)([一-鿿ァ-ヴー]{2,12})",
            text,
        ):
            obj = _clean_object(match.group(2))
            if obj:
                hint = "give" if match.group(1).startswith("あげ") else "receive"
                pairs.append((verb, obj, hint))
    # 同じ (動詞, 客体, ヒント) は 1 つに畳む（出現順を保つ）。
    return list(dict.fromkeys(pairs))


def _giving_hint(text: str, position: int) -> str:
    """指定位置（動詞の直後）にある授受表現から向きの手掛かりを返す。

    「作ってあげた」「買ってくれた」のように、授受表現はその行為の直後に付く。
    離れた位置の授受表現は別の行為に係っているので見ない。
    """
    tail = text[position : position + 10]
    if _GIVE_RE.search(tail):
        return "give"
    if _RECEIVE_RE.search(tail):
        return "receive"
    return ""


def _direction_from_giving(text: str, role: str) -> str:
    """授受表現と発話者(role)から行為の向きを決める。判定不能なら 'unknown'。

    日本語の授受表現は主客を明示する強い信号なので、これが取れた往復は
    LLM に回さずに確定できる。role が user か assistant かで向きが反転する点に注意。
    """
    gives = bool(_GIVE_RE.search(text))
    receives = bool(_RECEIVE_RE.search(text))
    if gives == receives:
        return "unknown"  # 両方 / どちらも無し → ここでは決めない
    if role == "user":
        return "user->char" if gives else "char->user"
    return "char->user" if gives else "user->char"


def _names_for(direction: str, user_name: str, char_name: str) -> tuple[str, str]:
    """向きから (主体, 受け手) の名前を割り当てる。総称語しか無ければ空文字。"""
    user = user_name if user_name not in _GENERIC_NAMES else ""
    char = char_name if char_name not in _GENERIC_NAMES else ""
    if direction == "user->char":
        return user, char
    if direction == "char->user":
        return char, user
    return "", ""


def extract_rule_based(
    user_text: str,
    reply_text: str,
    *,
    user_name: str = "",
    char_name: str = "",
    mode: str = "normal",
    speaker: str = "",
) -> list[dict]:
    """授受表現＋role のルールだけで事実を抽出する（LLM 呼び出しなし）。

    user 発話と assistant 返答の両方を見る。事実は往復のどちら側で語られても
    等しく事実なので、返答側にしか出てこない客体（「星の王子さまありがとう」）も拾う。
    向きが決まらなかった場合も verb / object が取れていれば
    ``direction='unknown'`` で返す（捨てない）。
    """
    user_text = str(user_text or "").strip()
    reply_text = str(reply_text or "").strip()
    user_name = str(user_name or "").strip()
    char_name = str(char_name or "").strip()
    two_only = str(mode or "").strip() == "two_only"
    facts: list[dict] = []
    sides = (("user", user_text), ("assistant", reply_text))
    for role, text in sides:
        if not text:
            continue
        # 動詞と客体は係り受けを保った組で取る（総当たりの組み合わせは誤りを量産する）。
        pairs = _find_verb_object_pairs(text)
        if not pairs:
            continue
        # 本文に明示された主体（「ナデシコが壊した」）。向きが取れないときの手掛かり。
        named_subject = ""
        match = _SUBJECT_RE.search(text)
        if match:
            candidate = _TIME_PREFIX_RE.sub("", match.group(1)).strip()
            if candidate and candidate not in _GENERIC_NAMES:
                named_subject = candidate
        # 向きは行為ごとに決める。文全体で1つに決めてしまうと、別の行為に係る
        # 授受表現の向きを持ち込む（実測: 「食べてくれるから…パンツを買いに行く」で
        # 買う行為が「ルリが買った」になった）。
        for verb, obj, hint in pairs:
            if not hint:
                direction = "unknown"
            elif role == "user":
                direction = "user->char" if hint == "give" else "char->user"
            else:
                direction = "char->user" if hint == "give" else "user->char"
            if two_only:
                # 2人だけモードにユーザーは存在しない。向きはキャラ間なので、
                # ユーザー主体の向きを持ち込まない（幻のユーザーを台帳へ作らない）。
                direction = "char->char" if direction != "unknown" else "unknown"
                subject = str(speaker or "").strip() if role == "assistant" else ""
                recipient = ""
            else:
                subject, recipient = _names_for(direction, user_name, char_name)
                if not subject:
                    subject = named_subject
            facts.append(
                {
                    "category": _guess_category(obj, verb),
                    "subject": subject,
                    "verb": verb,
                    "object": obj,
                    "recipient": recipient,
                    "direction": direction,
                    # ルールの確信度: その行為に係る授受表現で向きが取れた行を高くする。
                    "confidence": 0.85 if direction != "unknown" else 0.4,
                    "extractor": "rule",
                    "snippet": text,
                }
            )
    return _resolve_conflicts(_dedupe(facts))


def _resolve_conflicts(facts: list[dict]) -> list[dict]:
    """同じ（動詞・客体）に食い違う向きが付いた組を 1 件へ畳む。

    LLM は同じ出来事を両方向で返すことがある（実測: 「お嫁さんの役割を引き受ける」が
    オサム→ルリ と ルリ→オサム の 2 件で入った）。同じ往復の中で向きが矛盾しているなら
    少なくとも一方は誤りなので、確信度の高い方だけを残す。優劣が付かないときは
    どちらとも言えないので `unknown`（主客不明）へ落とす —— 誤った向きを残して
    「俺が君にした事」の絞り込みに混ぜるより、不明として提示する方が安全。
    """
    groups: dict[tuple, list[dict]] = {}
    for fact in facts:
        key = (str(fact.get("verb") or ""), str(fact.get("object") or ""))
        groups.setdefault(key, []).append(fact)
    result: list[dict] = []
    for items in groups.values():
        directions = {str(item.get("direction") or "unknown") for item in items}
        if len(directions) <= 1:
            result.extend(items)
            continue
        best = max(items, key=lambda item: float(item.get("confidence") or 0))
        top = float(best.get("confidence") or 0)
        if sum(1 for item in items if float(item.get("confidence") or 0) == top) > 1:
            best = dict(best)
            best["direction"] = "unknown"
            best["recipient"] = ""
        result.append(best)
    return result


def _dedupe(facts: list[dict]) -> list[dict]:
    """同一往復内の重複（同じ verb/object/direction）を確信度の高い方で畳む。"""
    merged: dict[tuple, dict] = {}
    for fact in facts:
        key = (
            str(fact.get("verb") or ""),
            str(fact.get("object") or ""),
            str(fact.get("direction") or ""),
        )
        current = merged.get(key)
        if current is None or float(fact.get("confidence") or 0) > float(
            current.get("confidence") or 0
        ):
            merged[key] = fact
    return list(merged.values())


def needs_llm(facts: list[dict]) -> bool:
    """ルール抽出の結果を見て、LLM 抽出へ回すべきかを判定する。

    回すのは次のいずれか。授受表現で向きが確定し、具体的な客体まで取れた往復は
    LLM に投げない（速度と再現性のため）。
      ・何も取れなかった
      ・向きが 1 つも決まらなかった
      ・客体が総称語（「本」「料理」）しか取れなかった
        → 「本を買ってあげた。星の王子さま」の書名のように、ルールでは
          助詞の直前しか見ないため具体名を取り逃す。ここは LLM の方が強い。
    """
    if not facts:
        return True
    if all(str(fact.get("direction") or "unknown") == "unknown" for fact in facts):
        return True
    concrete = [
        str(fact.get("object") or "").strip()
        for fact in facts
        if str(fact.get("object") or "").strip() not in _VAGUE_OBJECTS
        and str(fact.get("object") or "").strip()
    ]
    return not concrete


def build_llm_prompt(
    user_text: str,
    reply_text: str,
    *,
    user_name: str = "",
    char_name: str = "",
    mode: str = "normal",
    speaker: str = "",
) -> tuple[str, str]:
    """LLM 抽出用の (system, user) プロンプトを返す。

    出力は JSON 配列のみ。曖昧なら空配列を返させる（推測で埋めさせない）。
    主客を取り違えないことが目的なので、向きの定義を明示的に与える。
    """
    user_label = str(user_name or "ユーザー").strip() or "ユーザー"
    char_label = str(char_name or "キャラクター").strip() or "キャラクター"
    if str(mode or "").strip() == "two_only":
        world = (
            f"これは登場人物どうしの会話（2人だけモード）で、ユーザーは存在しません。"
            f"向き(direction)は必ず \"char->char\" か \"unknown\" にしてください。"
            f"返答した人物は「{str(speaker or char_label)}」です。"
        )
    else:
        world = (
            f"これは「{user_label}」（ユーザー）と「{char_label}」（キャラクター）の会話です。"
            f"向き(direction)は、{user_label} が {char_label} にしたことなら \"user->char\"、"
            f"{char_label} が {user_label} にしたことなら \"char->user\"、"
            f"自分自身のことなら \"self\"、判断できないなら \"unknown\" とします。"
        )
    system = (
        "あなたは会話ログから事実を抽出する抽出器です。JSON 配列だけを出力します。\n"
        f"{world}\n"
        "各要素は次のキーを持つオブジェクトにしてください:\n"
        "・category は物の種類で決める。『贈り物』は**相手に渡した物**にだけ使い、"
        "自分のために買った物は『衣類』『本』のように物の種類を書く。\n"
        '  {"category": "料理|本|衣類|贈り物|場所|音楽|飲み物|出来事など", '
        '"subject": "行為をした人の名前", "verb": "作る|買う|渡す|行く|言う|壊す など辞書形", '
        '"object": "行為の対象（何を）", "recipient": "相手の名前", '
        '"direction": "user->char|char->user|char->char|self|unknown", '
        '"confidence": 0.0〜1.0}\n'
        "規則:\n"
        "・実際に起きた具体的な出来事だけを書く。感想・意見・冗談・仮定・予定は書かない。\n"
        "・**後から「何があったか」を思い出す手がかりになる事実だけ**を書く。挨拶・相槌・"
        "呼びかけ・気持ちの表明など、内容の無い行為は書かない"
        "（悪い例: 言う→挨拶 / 言う→言葉 / 思う→気持ち / 見る→顔）。"
        "具体的な品目・作品名・場所・出来事が対象になるものだけを残す"
        "（良い例: 作る→肉じゃが / 買う→星の王子さま / 行く→水族館）。\n"
        "・object が『こと』『もの』『気』『話』のような中身の無い語になるなら、"
        "その事実は出力しない。\n"
        "・『言う』『見る』『思う』のような発話・知覚だけの行為は書かない"
        "（会話 log 自体が言ったことの記録なので冗長）。ただし約束・取り決め・告白・"
        "打ち明けた秘密のように、後から内容を参照する価値がある発話は"
        "その内容を object にして書く（例: 約束する→新婚旅行 / 打ち明ける→昔の失敗）。\n"
        "・誰がしたことなのかを絶対に取り違えない。分からなければ subject を空文字、"
        'direction を "unknown" にする（推測で埋めない）。\n'
        "・direction は「誰のためにした行為か」で決める。**相手のために／相手に対してした**と"
        "本文に書かれている場合だけ user->char・char->user にする。自分のためにした行為"
        "（自分の下着を買った、自分が走った、自分が風呂に入った）は必ず self にし、"
        "recipient は空にする。相手が同席していただけの行為を「相手にしてあげた」に"
        "しないこと。判断できないなら unknown。\n"
        "・同じ出来事を両方向で二重に出さない（user->char と char->user を同時に書かない）。"
        "どちらか一方に決められないなら unknown を 1 件だけ出す。\n"
        "・1 つの発話に複数の事実があれば複数要素にする。何も無ければ [] を返す。\n"
        "・object は「讃岐うどん」「星の王子さま」のように短い名詞にする。文をそのまま入れない。\n"
        "・説明・前置き・コードブロックは出さず、JSON 配列だけを返す。/no_think"
    )
    stimulus_label = "お題" if str(mode or "").strip() == "two_only" else user_label
    reply_label = str(speaker or char_label).strip() or char_label
    user = (
        f"{stimulus_label}の発言:\n{str(user_text or '').strip()}\n\n"
        f"{reply_label}の発言:\n{str(reply_text or '').strip()}\n\nJSON:"
    )
    return system, user


def parse_llm_facts(content: str, *, user_name: str = "", char_name: str = "") -> list[dict]:
    """LLM 出力から JSON 配列を取り出して事実の列へ整える。失敗時は空リスト。

    思考タグ・前置き・コードフェンスが混ざっても、最初の JSON 配列だけを拾う。
    向きの表記揺れ（"user->character" 等）はここで正規化する。
    """
    text = str(content or "").strip()
    if not text:
        return []
    # <think>…</think> や前置きを落とし、最初の [ … ] を取る。
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json)?|```", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return []
    if not isinstance(parsed, list):
        return []
    facts: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        direction = str(item.get("direction") or "unknown").strip().lower()
        direction = direction.replace("character", "char").replace(" ", "")
        direction = direction.replace("user->chars", "user->char")
        if direction not in {"user->char", "char->user", "char->char", "self"}:
            direction = "unknown"
        subject = str(item.get("subject") or "").strip()
        recipient = str(item.get("recipient") or "").strip()
        # 総称語で返ってきた主体・受け手は、向きから実名へ寄せる（台帳の絞り込みで
        # 「私」「君」が混ざると集計できないため）。
        if subject in _GENERIC_NAMES or recipient in _GENERIC_NAMES:
            named_subject, named_recipient = _names_for(
                direction, str(user_name or "").strip(), str(char_name or "").strip()
            )
            subject = subject if subject not in _GENERIC_NAMES else named_subject
            recipient = recipient if recipient not in _GENERIC_NAMES else named_recipient
        obj = _clean_object(item.get("object"))
        verb = str(item.get("verb") or "").strip()
        # 客体が中身の無い語（プロンプトで禁じているが LLM は時々返す）なら捨てる。
        # 動詞だけの事実は「何があったか」を示せないため台帳に入れない。
        if not obj:
            continue
        # 受け手が第三者（ユーザーでもこのキャラでもない）なら向きは unknown にする。
        # 「オサム→ナデシコ 届ける」を user->char のままにすると、「俺が君にした事」の
        # 絞り込みに第三者への行為が混ざる。事実自体は有用なので捨てず、主客不明として
        # 残す（提示時に「（主客不明）」と明示される）。
        known_names = {
            name
            for name in (str(user_name or "").strip(), str(char_name or "").strip())
            if name
        }
        if direction in {"user->char", "char->user"} and known_names:
            involved = {name for name in (subject, recipient) if name}
            if involved and not involved <= known_names:
                direction = "unknown"
        # self（自分自身のこと）に受け手がいるのは矛盾なので受け手を落とす。
        if direction == "self":
            recipient = ""
        try:
            confidence = float(item.get("confidence") or 0.6)
        except (TypeError, ValueError):
            confidence = 0.6
        facts.append(
            {
                "category": str(item.get("category") or "").strip()
                or _guess_category(obj, verb),
                "subject": subject,
                "verb": verb,
                "object": obj,
                "recipient": recipient,
                "direction": direction,
                "confidence": max(0.0, min(1.0, confidence)),
                "extractor": "llm",
                "snippet": "",
            }
        )
    return _resolve_conflicts(_dedupe(facts))


def infer_query_filters(
    question: str, *, user_name: str = "", char_name: str = ""
) -> dict:
    """質問文から台帳を引くための絞り込み条件を推定する。

    「俺が君に作ってあげた料理は？」→ ``{'verb': '作る', 'direction': 'user->char',
    'category': '料理'}``。「君が俺に作ってくれた料理は？」なら direction が反転する。
    ここで向きを取り違えると想起も間違うため、判定は抽出時と同じ授受表現ルールを使う
    （質問は user の発話なので role='user' で解釈する）。
    決められなかった項目は空文字にして、呼び出し側で絞り込みを緩める。
    """
    text = str(question or "").strip()
    if not text:
        return {"category": "", "verb": "", "direction": "", "subject": ""}
    verbs = _find_verbs(text)
    verb = verbs[0] if verbs else ""
    direction = _direction_from_giving(text, "user")
    if direction == "unknown":
        # 授受表現が無い質問（「作った料理は？」）。この会話は基本ユーザー→キャラの
        # 関係なので、主体が明示されていなければ user->char を既定にする。
        # 「君が」「あなたが」のようにキャラ主体が明示された場合は反転させる。
        char = str(char_name or "").strip()
        if re.search(r"(?:君|きみ|あなた|お前|そっち)が", text) or (
            char and re.search(re.escape(char) + r"が", text)
        ):
            direction = "char->user"
        elif re.search(r"(?:俺|おれ|私|わたし|僕|ぼく|自分)が", text) or (
            str(user_name or "").strip()
            and re.search(re.escape(str(user_name).strip()) + r"が", text)
        ):
            direction = "user->char"
        else:
            direction = ""  # 決めない（呼び出し側は向き無しで引いて両方見せる）
    category = ""
    for name, pattern in _CATEGORY_HINTS:
        if re.search(pattern, text):
            category = name
            break
    return {
        "category": category,
        "verb": verb,
        "direction": direction if direction != "unknown" else "",
        "subject": "",
    }


def extract(
    user_text: str,
    reply_text: str,
    *,
    user_name: str = "",
    char_name: str = "",
    mode: str = "normal",
    speaker: str = "",
    llm=None,
) -> list[dict]:
    """1 往復から事実を抽出する（ルール → 必要なら LLM のハイブリッド）。

    ``llm`` は ``llm(system: str, user: str) -> str`` を満たす callable。
    None ならルールだけで抽出する（LM Studio が落ちていても台帳構築は進む）。
    LLM 側が失敗・空・パース不能でも、ルールの結果をそのまま返す。
    """
    facts = extract_rule_based(
        user_text,
        reply_text,
        user_name=user_name,
        char_name=char_name,
        mode=mode,
        speaker=speaker,
    )
    if llm is None or not needs_llm(facts):
        return facts
    system, prompt = build_llm_prompt(
        user_text,
        reply_text,
        user_name=user_name,
        char_name=char_name,
        mode=mode,
        speaker=speaker,
    )
    try:
        content = llm(system, prompt)
    except Exception:
        return facts
    llm_facts = parse_llm_facts(content, user_name=user_name, char_name=char_name)
    if not llm_facts:
        return facts
    # LLM の結果を優先しつつ、ルールが取れていた事実も残す（和集合）。
    # 同じ verb/object/direction は _dedupe が確信度の高い方を採る。
    for fact in llm_facts:
        if not str(fact.get("snippet") or "").strip():
            fact["snippet"] = str(user_text or "").strip()
    # ただしルール側の弱い事実（向きが決まらなかったもの）は捨てる。
    # ルールの客体抽出は「を」の直前を取る素朴な規則なので、主客も客体も曖昧な行は
    # 台帳のノイズにしかならない（実データで「そしてうまい晩飯」「後日結果」のような
    # 切り出し失敗が混ざった）。LLM が答えを出せているなら弱い推測は要らない。
    strong_rule = [
        fact for fact in facts if str(fact.get("direction") or "unknown") != "unknown"
    ]
    return _resolve_conflicts(_dedupe(llm_facts + strong_rule))
