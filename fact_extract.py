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
    ("見る", r"見(?:た|て|せ)|観(?:た|て)|眺め"),
    ("言う", r"言(?:っ|う)|伝え|教え"),
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
    (
        "料理",
        r"料理|ご飯|ごはん|飯|おかず|晩御飯|晩ごはん|夕飯|朝食|昼食|弁当|"
        r"うどん|そば|ラーメン|カレー|シチュー|パスタ|スパゲ|餃子|チャーハン|"
        r"味噌汁|スープ|サラダ|煮物|炒め|天ぷら|唐揚げ|卵焼き|オムレツ|ハンバーグ|"
        r"肉じゃが|麻婆|丼|寿司|刺身|焼き魚|塩じゃけ|鮭|さけ|たたき|鍋|おでん|"
        r"ケーキ|クッキー|プリン|パン|ピザ|グラタン|コロッケ|春巻|パエリア",
    ),
    ("飲み物", r"コーヒー|紅茶|お茶|ジュース|ココア|ミルク|ビール|ワイン|水|スムージー"),
    ("贈り物", r"プレゼント|贈り物|花束|指輪|ネックレス|ぬいぐるみ|人形|服|靴|鞄|かばん"),
    ("音楽", r"歌|曲|音楽|ピアノ|ギター|アルバム|ライブ"),
    ("場所", r"公園|海|山|川|映画館|水族館|動物園|遊園地|神社|寺|喫茶店|カフェ|レストラン|旅行"),
)
# 授受表現（方向の決定に使う）。発話者から相手への行為か、その逆か。
# 「〜てあげる／やる」は発話者→相手、「〜てくれる／もらう／いただく」は相手→発話者。
_GIVE_RE = re.compile(r"(?:て|で)(?:あげ|やっ|やる|あげる)|してあげ|プレゼントし")
_RECEIVE_RE = re.compile(r"(?:て|で)(?:くれ|もらっ|もらう|いただ)|してくれ|もらった")
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
    """抽出した客体から前置きの助詞・連体修飾を落として名詞句へ寄せる。"""
    body = str(text or "").strip()
    body = _OBJECT_TRIM_RE.sub("", body).strip()
    body = _OBJECT_MODIFIER_RE.sub("", body).strip()
    body = body.strip("、。・…「」『』\"'`（）() 　")
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
    """本文に現れる行為の動詞（正規化後）を出現順で返す。"""
    found: list[tuple[int, str]] = []
    for verb, pattern in _VERB_PATTERNS:
        match = re.search(pattern, text)
        if match:
            found.append((match.start(), verb))
    found.sort()
    return [verb for _pos, verb in found]


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
        verbs = _find_verbs(text)
        if not verbs:
            continue
        direction = _direction_from_giving(text, role)
        if two_only:
            # 2人だけモードにユーザーは存在しない。向きはキャラ間なので、
            # ユーザー主体の向きを持ち込まない（幻のユーザーを台帳へ作らない）。
            direction = "char->char" if direction != "unknown" else "unknown"
            subject = str(speaker or "").strip() if role == "assistant" else ""
            recipient = ""
        else:
            subject, recipient = _names_for(direction, user_name, char_name)
            if not subject:
                # 向きが不明でも、主体が本文に明示されていれば拾う（「ナデシコが壊した」）。
                match = _SUBJECT_RE.search(text)
                if match:
                    candidate = _TIME_PREFIX_RE.sub("", match.group(1)).strip()
                    if candidate and candidate not in _GENERIC_NAMES:
                        subject = candidate
        objects = [_clean_object(m) for m in _OBJECT_RE.findall(text)]
        # 「作ってくれたオムライス」のように「を」を伴わない客体も拾う。
        objects += [_clean_object(m) for m in _OBJECT_AFTER_RE.findall(text)]
        objects = [obj for obj in objects if obj]
        # 同じ客体が両経路で取れることがあるので、順序を保って重複を除く。
        objects = list(dict.fromkeys(objects))
        for verb in verbs:
            targets = objects or [""]
            for obj in targets:
                if not obj and direction == "unknown":
                    continue  # 客体も向きも無い＝情報が無いので台帳へ入れない
                facts.append(
                    {
                        "category": _guess_category(obj, verb),
                        "subject": subject,
                        "verb": verb,
                        "object": obj,
                        "recipient": recipient,
                        "direction": direction,
                        # ルールの確信度: 授受表現で向きが取れた行を高くする。
                        "confidence": 0.85 if direction != "unknown" else 0.4,
                        "extractor": "rule",
                        "snippet": text,
                    }
                )
    return _dedupe(facts)


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
        '  {"category": "料理|本|贈り物|場所|音楽|飲み物|出来事など", '
        '"subject": "行為をした人の名前", "verb": "作る|買う|渡す|行く|言う|壊す など辞書形", '
        '"object": "行為の対象（何を）", "recipient": "相手の名前", '
        '"direction": "user->char|char->user|char->char|self|unknown", '
        '"confidence": 0.0〜1.0}\n'
        "規則:\n"
        "・実際に起きた具体的な出来事だけを書く。感想・意見・冗談・仮定・予定は書かない。\n"
        "・誰がしたことなのかを絶対に取り違えない。分からなければ subject を空文字、"
        'direction を "unknown" にする（推測で埋めない）。\n'
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
        obj = str(item.get("object") or "").strip()
        verb = str(item.get("verb") or "").strip()
        if not obj and not verb:
            continue
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
    return _dedupe(facts)


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
    return _dedupe(llm_facts + facts)
