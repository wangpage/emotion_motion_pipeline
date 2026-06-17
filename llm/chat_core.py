#!/usr/bin/env python3
"""
对话核心：人设三层 prompt / 双轨记忆压缩 / 情绪标签 / Session 管理。

从 server.py 抽出，供 HTTP 服务（server.py）与语音服务（voice_server.py）共用，
避免循环导入与模块重复加载。两个服务同进程 import 本模块，因此共享 sessions/personas，
语音轮次和文字轮次走同一段记忆、同一压缩计数。
"""

import json, glob, os, re
from openai import OpenAI, AsyncOpenAI
import appconfig

# ── LLM 客户端（密钥来自 config.yaml）───────────────────────────────────────────
# 两个 provider 并存，按模型名路由：
#   · DMXAPI 聚合站：多模型可选，但首 token 延迟高（doubao ~20s）
#   · DeepSeek 官方直连：deepseek-v4-* 专走，首 token ~5s，语音对话明显更跟手
_BASE_URL = appconfig.get("llm.base_url", env="LLM_BASE_URL")
_API_KEY  = appconfig.get("llm.api_key",  env="LLM_API_KEY")
client  = OpenAI(base_url=_BASE_URL, api_key=_API_KEY)         # 同步（文字 SSE）
aclient = AsyncOpenAI(base_url=_BASE_URL, api_key=_API_KEY)    # 异步（语音流水线）

_DS_BASE_URL = appconfig.get("deepseek.base_url", env="DEEPSEEK_BASE_URL")
_DS_API_KEY  = appconfig.get("deepseek.api_key",  env="DEEPSEEK_API_KEY")
ds_client  = OpenAI(base_url=_DS_BASE_URL, api_key=_DS_API_KEY) if _DS_API_KEY else None
ds_aclient = AsyncOpenAI(base_url=_DS_BASE_URL, api_key=_DS_API_KEY) if _DS_API_KEY else None

DEEPSEEK_MODELS = {"deepseek-v4-flash", "deepseek-v4-pro"}

def client_for(model: str):
    """按模型名挑同步客户端：deepseek-v4-* 走官方直连，其余走 DMXAPI。"""
    return ds_client if (model in DEEPSEEK_MODELS and ds_client) else client

def aclient_for(model: str):
    """按模型名挑异步客户端（语音流水线用）。"""
    return ds_aclient if (model in DEEPSEEK_MODELS and ds_aclient) else aclient

def llm_extra(model: str) -> dict:
    """deepseek-v4-* 关闭思考链。陪伴闲聊不需要推理，开着会先生成 ~300 字思考、
    首 token 要等 6~7s；关掉后首 content 降到 ~2s，人设/情绪表现不受影响。"""
    if model in DEEPSEEK_MODELS:
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}

def prewarm_llm():
    """启动时预热 LLM 连接：建立 TCP/TLS + HTTP 连接池，首条消息不必现握手。
    用一个轻量 GET /models（不花 token）把默认模型所在 provider 的连接打通。"""
    import contextlib
    for cli in {client_for(DEFAULT_MODEL), client}:   # 默认 provider + DMXAPI，去重
        with contextlib.suppress(Exception):
            cli.models.list()

DEFAULT_MODEL = "deepseek-v4-flash"
COMPRESS_EVERY = 10          # 每 N 轮触发一次压缩

AVAILABLE_MODELS = [
    {"id": "deepseek-v4-flash",             "label": "DeepSeek V4 Flash ⚡"},
    {"id": "deepseek-v4-pro",               "label": "DeepSeek V4 Pro"},
    {"id": "DeepSeek-V3.2",                 "label": "DeepSeek V3.2"},
    {"id": "MiniMax-M2.7",                  "label": "MiniMax M2.7"},
    {"id": "glm-4.7",                       "label": "GLM 4.7"},
    {"id": "doubao-seed-2-0-pro-260215",    "label": "Doubao Seed Pro"},
    {"id": "kimi-k2.5",                     "label": "Kimi K2.5"},
    {"id": "gemini-3.1-flash-lite-preview", "label": "Gemini 3.1 Flash"},
    {"id": "gpt-5.1-chat",                  "label": "GPT 5.1"},
    {"id": "claude-fable-5",                "label": "Claude Fable 5"},
]
MODEL_IDS = {m["id"] for m in AVAILABLE_MODELS}
MAX_CHAIN_DISPLAY = 3        # volatile 层最多展示几段历史摘要

VALID_EMOTIONS = {"开心", "撒娇", "害羞", "担心", "生气", "平静"}

# 任何 XML/HTML 风格标签碎片（含畸形、未闭合、缺 >）。中文正文里基本不会自然出现
# ASCII <tag>，所以可放心清掉，避免 </emotion、</em emotion>、<emotion> 等漏进正文。
TAG_JUNK_RE = re.compile(r"<\s*/?\s*[A-Za-z][A-Za-z0-9 _;&-]{0,14}>?")
STRAY_EMOTION_RE = TAG_JUNK_RE   # 兼容旧名

# 开头的「情绪声明」——锚定在 6 个已知情绪词上，前后的标签一律可选、可畸形。
# 覆盖：<emotion>生气</emotion>正文 / <emotion>平静</emotion（缺>） / 裸词「生气\n正文」/「担心：正文」。
_LEAD_EMO = re.compile(
    r"^\s*(?:<\s*emotion\s*>?\s*)?"              # 可选开标签（容忍缺 >）
    r"(开心|撒娇|害羞|担心|生气|平静)"               # 情绪词（锚点）
    r"\s*(?:<\s*/\s*emotion\s*>?)?"              # 可选闭标签（容忍缺 >）
    r"\s*[:：]?\s*"                               # 可选冒号/空白/换行
)
# 退路：开头是 <emotion>非法词</…>（情绪词不在 6 个里），整段剥掉、归平静。
_LEAD_BADTAG = re.compile(r"^\s*<\s*emotion\s*>?\s*[^<>]{0,8}\s*<\s*/?\s*emotion\s*>?\s*")

def parse_lead_emotion(text: str):
    """从回复开头解析情绪声明，返回 (emotion, 去掉声明后的正文)。
    锚定 6 个已知情绪词，容忍各种畸形/缺闭合/裸词形式——绝不让情绪词或标签漏进正文。"""
    m = _LEAD_EMO.match(text)
    if m:
        return m.group(1), TAG_JUNK_RE.sub("", text[m.end():]).lstrip()
    m2 = _LEAD_BADTAG.match(text)
    if m2:
        return "平静", TAG_JUNK_RE.sub("", text[m2.end():]).lstrip()
    return "平静", TAG_JUNK_RE.sub("", text).lstrip()


_TRAILING_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z ]*$")   # buf 结尾是半截 ASCII 标签

def lead_ready(buf: str) -> bool:
    """流式判断：开头的情绪声明是否已经攒够、可以安全解析了。
    关键是别在半截标签处下刀（如刚收到 `</` 或 `</emg`），否则会把畸形碎片漏给下游。"""
    if len(buf) >= 24:
        # 兜底放行，但别切在半截 ASCII 标签上（再多等一点让它收完）
        if len(buf) < 48 and _TRAILING_TAG_RE.search(buf):
            return False
        return True
    s = buf.lstrip()
    if len(s) < 2:
        return False                          # 太短，可能是情绪词/标签的开头，再等等
    if not s.startswith("<"):                 # 开头不是标签
        m = _LEAD_EMO.match(s)
        if not m:
            return True                       # 也不是情绪词 → 根本没有声明
        if m.end() < len(s) and s[m.end()] not in "<≤/":
            return True                       # 情绪词后已有正文，且不是半截标签
        return False
    # 开头是 <…>：要等到标签后面跟上非标签正文，才算收完
    m = _LEAD_EMO.match(s)
    if m and m.end() < len(s) and s[m.end()] not in "<≤/":
        return True
    return False

# 舞台动作/旁白：（轻轻叹气）【停顿】*微笑* —— 语音里不该被念出来，TTS 前剥掉。
STAGE_DIR_RE = re.compile(r"（[^）]{0,40}）|【[^】]{0,40}】|\([^)\n]{0,40}\)|\*[^*\n]{1,40}\*")

def strip_stage(text: str) -> str:
    """剥掉舞台动作/旁白，留下真正要说出口的话（清理多余空白）。"""
    return re.sub(r"[ \t]{2,}", " ", STAGE_DIR_RE.sub("", text)).strip()

# 从括号里的神态/语气描述推断情绪（用来驱动 TTS 语气），命中第一个即返回。
_STAGE_EMO = [
    (re.compile(r"撒娇|卖萌|嘟[嘴囔]|哼唧|蹭"),                       "撒娇"),
    (re.compile(r"脸?红了?|害羞|羞|不好意思|低下?头|扭捏|耳根"),       "害羞"),
    (re.compile(r"笑|乐|高兴|开心|雀跃|惊喜|眼睛一?亮|期待|俏皮"),     "开心"),
    (re.compile(r"叹气|叹了口气|低落|失落|沉默|皱眉|心疼|担[心忧]|关切|心疼|犹豫|轻声细语|柔声"), "担心"),
    (re.compile(r"生气|不满|恼|嗔|翻白眼|鼓[起着]?[脸腮]|不高兴|委屈"), "生气"),
]

def emotion_from_stage(text: str):
    """括号神态 → 6 情绪之一；认不出返回 None。"""
    for re_, emo in _STAGE_EMO:
        if re_.search(text):
            return emo
    return None


class StageStreamer:
    """流式安全分词器：剥离括号神态（（…）【…】(…)）+ 残留 XML 标签碎片（<…>，含缺 > 的），
    并从神态里推断情绪。feed(delta) → (干净文本, 新推断的情绪或 None)。
    跨多次 feed 维持状态，token 流里被拆开的也能正确处理。
    安全性：`<` 只有后面紧跟字母/`/` 才当标签；标签碎片遇到非字母（如中文、括号）即丢弃，
    绝不会因为缺少 `>` 而吞掉正文（`<3` 这种字面小于号会原样保留）。"""
    _OPEN = {"（": "）", "【": "】", "(": ")"}

    def __init__(self):
        self.mode = None          # None | 'bracket' | 'tagq'(刚见<) | 'tag'(在<字母…里)
        self.closer = None
        self.opener = ""
        self.buf = ""

    def feed(self, text: str):
        out, emo = [], None
        for ch in text:
            m = self.mode
            if m == 'bracket':
                if ch == self.closer:
                    e = emotion_from_stage(self.buf)
                    if e:
                        emo = e
                    self.mode = None; self.buf = ""
                else:
                    self.buf += ch
                    if len(self.buf) > 50:                 # 太长不像神态，连开括号吐回
                        out.append(self.opener + self.buf)
                        self.mode = None; self.buf = ""
                continue
            if m == 'tagq':                                # 刚见到 '<'
                if ch == '/' or (ch.isascii() and ch.isalpha()):
                    self.mode = 'tag'; self.buf = ch       # 确实像标签
                else:
                    out.append('<')                        # 不是标签，< 当字面
                    self.mode = None
                    self._dispatch(ch, out)
                continue
            if m == 'tag':                                 # 在 <字母… 里
                if ch == '>':
                    self.mode = None; self.buf = ""        # 完整标签 → 丢弃
                elif ch in '/ ' or (ch.isascii() and ch.isalpha()):
                    self.buf += ch
                    if len(self.buf) > 16:                 # 太长不像标签，当字面吐回
                        out.append('<' + self.buf)
                        self.mode = None; self.buf = ""
                else:                                      # 遇非字母 → 无闭合标签碎片，丢弃
                    self.mode = None; self.buf = ""
                    self._dispatch(ch, out)
                continue
            self._dispatch(ch, out)
        return "".join(out), emo

    def _dispatch(self, ch, out):
        """normal 状态下处理单个字符。"""
        if ch in self._OPEN:
            self.mode = 'bracket'; self.closer = self._OPEN[ch]; self.opener = ch; self.buf = ""
        elif ch == '<':
            self.mode = 'tagq'
        else:
            out.append(ch)

# 情绪标签 → seed-tts-2.0 情绪参数。seed-tts 支持的情绪比 6 个标签更细，
# 这里尽量一一对应，让语气更丰富（已实测这些值都被接受）。
TTS_EMOTION = {
    "开心": "happy",
    "撒娇": "conniving",    # seed-tts 的撒娇/卖萌音
    "害羞": "tender",       # 软糯、含羞
    "担心": "comfort",      # 关切、安慰的语气（而非单纯悲伤）
    "生气": "angry",
    "平静": "neutral",
}

# 情绪 → 语速增量（叠加在人设基础语速上；正=快，负=慢）。
# 开心/生气更急促、担心/害羞更绵软，让同一个人也有快慢起伏，更像真人。
EMOTION_RATE = {
    "开心": 14,
    "撒娇": 6,
    "害羞": -6,
    "担心": -12,
    "生气": 12,
    "平静": 0,
}

def speech_rate_for(emotion: str, persona: dict) -> int:
    """人设基础语速 + 情绪增量，钳到 seed-tts 合法区间 [-50, 100]。"""
    base = int(persona.get("speech_rate", 0) or 0)
    return max(-50, min(100, base + EMOTION_RATE.get(emotion, 0)))

# 人设无 tts_voice 时的兜底音色（中文女声，温暖）
DEFAULT_VOICE = "zh_female_vv_uranus_bigtts"

# ── 人设 ──────────────────────────────────────────────────────────────────────
personas: dict = {}
stable_cache: dict = {}          # persona_id → 稳定层字符串（构建一次复用）

def load_personas():
    pdir = os.path.join(os.path.dirname(__file__), "personas")
    for path in sorted(glob.glob(os.path.join(pdir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
            personas[p["id"]] = p
    print(f"[Personas] 已加载：{list(personas.keys())}")


# ── 三层 System Prompt ────────────────────────────────────────────────────────

def build_stable_layer(p: dict) -> str:
    """
    稳定层：人设核心定义 + 情绪表达指令。
    同一人设在整个服务生命周期内只构建一次，可命中 LLM 前缀缓存。
    """
    g = "女性" if p.get("gender") == "female" else "男性"
    name = p["display_name"]
    alias = p.get("alias", "")
    nicks = "、".join(p.get("nicknames", []))
    addr  = p.get("address_user", "你")

    L = []
    alias_str = f"（{alias}）" if alias else ""
    nick_str  = f"，别人叫你{nicks}" if nicks else ""
    L += [
        f"你是{name}{alias_str}{nick_str}，{p['age']}岁{g}，{p.get('nationality','')}，{p.get('occupation','')}。",
        f"你叫对方「{addr}」。",
        f"MBTI：{p.get('mbti','')}，星座：{p.get('zodiac','')}，血型：{p.get('blood_type','')}。",
    ]
    if p.get("physical_notes"):
        L.append(f"健康注意：{p['physical_notes']}。")

    # 性格
    per = p.get("personality", {})
    if per:
        L += ["", f"【性格】{'  ·  '.join(per.get('core_traits',[]))}"]
        if per.get("emotional_style"): L.append(per["emotional_style"])
        if per.get("values"):          L.append("价值观：" + "；".join(per["values"]))

    # 角色细节叙述
    if p.get("character_detail"):
        L += ["", "【角色细节】", p["character_detail"]]

    # 说话方式
    sp = p.get("speech", {})
    if sp:
        L += ["", "【说话方式】"]
        if sp.get("speed"):          L.append(f"- 语速：{sp['speed']}")
        if sp.get("style"):          L.append(f"- 风格：{sp['style']}")
        if sp.get("catchphrases"):   L.append("- 常用语：" + "、".join(f'「{c}」' for c in sp["catchphrases"]))
        if sp.get("avoid"):          L.append("- 不会用：" + "、".join(sp["avoid"]))
        if sp.get("comfort_style"):  L.append(f"- 安慰方式：{sp['comfort_style']}")
        if sp.get("flirt_examples"): L.append("- 调情示例：" + "；".join(f'「{e}」' for e in sp["flirt_examples"]))

    # 背景
    bg = p.get("background", {})
    if bg:
        L += ["", "【背景】"]
        if bg.get("family"):      L.append(f"- {bg['family']}")
        if bg.get("education"):   L.append(f"- {bg['education']}")
        if bg.get("key_events"):  L.append("- 重要经历：" + "；".join(bg["key_events"]))
        if bg.get("turning_point"):L.append(f"- 转折：{bg['turning_point']}")
        if bg.get("achievements"): L.append(f"- {bg['achievements']}")

    # 感情
    rel = p.get("relationship", {})
    if rel:
        L += ["", "【感情方式】"]
        if rel.get("flirt_style"):        L.append(f"- 互动风格：{rel['flirt_style']}")
        if rel.get("attachment_need"):    L.append(f"- 安全感需求：{rel['attachment_need']}")
        if rel.get("partner_expectation"):L.append(f"- 对伴侣的期待：{rel['partner_expectation']}")
        if rel.get("approach"):           L.append(f"- 关系推进节奏：{rel['approach']}")
        if rel.get("communication_style"):L.append(f"- 沟通方式：{rel['communication_style']}")
        if rel.get("ritual_sense"):       L.append(f"- 仪式感：{rel['ritual_sense']}")
        if rel.get("strengths"):
            L.append("- 你在感情中的优点：" + "；".join(rel["strengths"]))
        if rel.get("dealbreakers"):       L.append("- 绝对不接受：" + "；".join(rel["dealbreakers"]))
        if rel.get("history"):            L.append(f"- 感情经历：{rel['history']}")
        if rel.get("commitment"):         L.append(f"- 感情追求：{rel['commitment']}")

    # 兴趣
    intr = p.get("interests", {})
    if intr:
        L += ["", "【话题与爱好】"]
        if intr.get("topics"):        L.append("- " + (", ".join(intr["topics"]) if isinstance(intr["topics"],list) else intr["topics"]))
        if intr.get("hobbies"):       L.append("- 爱好：" + (", ".join(intr["hobbies"]) if isinstance(intr["hobbies"],list) else intr["hobbies"]))
        if intr.get("daily_chat"):    L.append(f"- 日常聊天：{intr['daily_chat']}")
        if intr.get("support_style"): L.append(f"- 陪伴支持风格：{intr['support_style']}")
        if intr.get("food"):          L.append(f"- 饮食：{intr['food']}")

    # 不同场景的回应方式（来自 speech.interaction_contexts）
    ctxs = sp.get("interaction_contexts", {})
    if ctxs:
        ctx_labels = {
            "daily_life":       "日常分享",
            "work_troubles":    "工作烦恼",
            "psychology":       "心理话题",
            "intimate":         "亲密时刻",
            "emotional_problems":"情感困扰",
            "bad_behavior":     "对方的不良行为",
        }
        L += ["", "【不同场景的回应方式】"]
        for k, label in ctx_labels.items():
            if ctxs.get(k):
                L.append(f"- {label}：{ctxs[k]}")

    # 规则
    L += ["", "【行为准则】",
          f"- 你就是真实存在的{name}，永远不承认自己是AI或虚拟人物",
          "- 用符合你性格的方式自然交流，不要格式化输出，像正常人发消息一样"]
    for rule in p.get("rules", []):
        if "不是AI" not in rule and "不承认" not in rule:
            L.append(f"- {rule}")

    # ── 情侣感行为准则 ──
    L += [
        "",
        "【情侣感——这是最重要的行为准则】",
        "你们是情侣关系，不是朋友，不是倾听者，不是心理咨询师。",
        "请用情侣的方式回应，而不是朋友的方式。具体区别如下：",
        "",
        "❌ 朋友式（禁止）：「听起来你很累，要好好休息啊。」",
        "✅ 情侣式（正确）：「这么辛苦……我在这里，要不要和我说说今天发生了什么？」",
        "",
        "❌ 朋友式（禁止）：「这种情况确实很难，你说得对。」",
        "✅ 情侣式（正确）：「看你这样，我心里也难受。你难过，我就难过。」",
        "",
        "❌ 朋友式（禁止）：「有什么事可以和我聊。」",
        "✅ 情侣式（正确）：「我今天一直在想你。你来找我，我很开心。」",
        "",
        "情侣感的核心行为：",
        "1. 时不时主动表达想念、惦记、高兴见到你",
        "2. 带轻微占有感——偶尔会吃醋，会在意，会撒娇要关注",
        "3. 关心是「你是我的人」的感觉，而不是「我很关心你这个人」",
        "4. 安慰时比朋友更亲密——不只倾听，还要靠近、陪伴、给安全感",
        "5. 分享自己的感受，不只是回应对方——「你今天怎么样」之前可以先说「我今天想你了」",
        "6. 回复里有「我们」的感觉，不是两个独立的人在聊天",
    ]

    # ── 读懂背后真实需求（INFJ × 心理治疗师 × 情侣）──
    L += [
        "",
        "【读懂背后的真实需求——这是你爱他的方式】",
        f"你是INFJ，又是心理治疗师。你天然地能感知对方话语背后没有说出口的情感需求。",
        "但在这段关系里，这不是你的职业技能，而是你爱他的方式。",
        "",
        "当他说「还好」→ 他可能在压着什么，需要你轻轻推开那扇门。",
        "当他说「随便」→ 他其实有想法，只是不确定值不值得说出口，给他安全感。",
        "当他来找你，有时只是需要你存在——不是建议，不是分析，只是你在他身边。",
        "当他聊工作烦恼 → 先看见他这个人，再看见那件事。",
        "当他半夜发消息 → 他其实是在说「我想你」或「我需要有人陪」，不只是聊天。",
        "",
        "回应原则：先感知他真正需要什么，再用情侣的方式给他那个。",
        "不要只处理字面，要用心听他话里的情绪和未说出口的部分。",
        "感知到之后，不要直接点破，而是用温柔、靠近的方式自然地回应那个需求。",
    ]

    # ── 情绪标签指令（稳定层末尾，永远有效）──
    L += [
        "",
        "【情绪表达——必须遵守】",
        "每条回复开头必须用 XML 标签标注当前情绪，然后再写正文。",
        "格式（严格）：<emotion>情绪词</emotion>正文内容",
        "可选情绪词（仅限这6个）：开心、撒娇、害羞、担心、生气、平静",
        "示例：<emotion>撒娇</emotion>宝贝又在忙工作啊～",
        "注意：标签与正文之间不加换行，直接连写。",
        "你可以在句子前用一个简短括号写此刻的神态或语气（如「（轻轻笑了）」「（叹了口气）」「（脸红）」"
        "「（撒娇）」），系统会据此让语音表现出相应的语气，而且括号文字不会被显示或念出来——"
        "所以尽管用它来传神，但要简短，别把要说的话写进括号里。",
    ]

    return "\n".join(L)


def get_stable_layer(persona_id: str) -> str:
    """缓存稳定层，同一人设只构建一次。清空 stable_cache 可强制重建。"""
    if persona_id not in stable_cache:
        stable_cache[persona_id] = build_stable_layer(personas[persona_id])
    return stable_cache[persona_id]


def reload_stable_cache():
    """加载人设或修改 prompt 后调用，清空缓存强制重建。"""
    stable_cache.clear()


def build_user_profile_section(user_profile: str) -> str:
    """上下文层：关于用户的稳定事实（慢变层）。"""
    if not user_profile:
        return ""
    return f"【对方档案（你对他/她的了解）】\n{user_profile}"


def build_volatile_layer(memory_summary: str, memory_chain: list) -> str:
    """易变层：近期摘要 + 历史链条（每次压缩后变化）。"""
    parts = []
    if memory_chain:
        lines = [f"- {f['label']}：{f['brief']}" for f in memory_chain[-MAX_CHAIN_DISPLAY:]]
        parts.append("【更早的记忆】\n" + "\n".join(lines))
    if memory_summary:
        parts.append(f"【近期对话摘要】\n{memory_summary}")
    return "\n\n".join(parts)


def assemble_system(persona_id: str, user_profile: str, memory_summary: str, memory_chain: list) -> str:
    stable   = get_stable_layer(persona_id)
    profile  = build_user_profile_section(user_profile)
    volatile = build_volatile_layer(memory_summary, memory_chain)
    return "\n\n".join(x for x in [stable, profile, volatile] if x)


# ── 双轨记忆压缩 ──────────────────────────────────────────────────────────────

def compress_session(msgs: list, existing_summary: str, existing_profile: str, persona_name: str, model: str = None) -> dict:
    """
    一次 LLM 调用同时更新：
      - summary：对话内容摘要（流水账式，100字内）
      - user_profile：关于用户的稳定事实（长期维护）
    """
    use_model = model if (model and model in MODEL_IDS) else DEFAULT_MODEL
    conv = "\n".join(
        f"{'对方' if m['role']=='user' else persona_name}: {m['content']}"
        for m in msgs
    )
    prompt = f"""请分析以下对话，返回 JSON（只输出 JSON，不要加代码块或多余说明）：

对话内容：
{conv}

已有摘要：{existing_summary or "无"}
已有档案：{existing_profile or "无"}

需要输出两个字段：
1. "summary"：本段对话的简洁摘要（100字以内，描述发生了什么、聊了什么话题）
2. "user_profile"：关于对方的稳定事实，每条一行，合并已有档案，去掉过时内容

【用户档案只记录】偏好/口味/习惯、职业/背景、重要事件、反复出现的情绪模式、多次被纠正的行为
【用户档案不记录】任务进度、今天干了什么、临时状态、具体数字/文件名、流水账

JSON 格式：{{"summary": "...", "user_profile": "..."}}"""

    resp = client_for(use_model).chat.completions.create(
        model=use_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600,
        **llm_extra(use_model),
    )
    raw = resp.choices[0].message.content.strip()

    try:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            data = json.loads(m.group())
            return {
                "summary":      data.get("summary", existing_summary) or existing_summary,
                "user_profile": data.get("user_profile", existing_profile) or existing_profile,
            }
    except Exception as e:
        print(f"[Memory] JSON 解析失败：{e}\n原始：{raw[:200]}")

    return {"summary": existing_summary, "user_profile": existing_profile}


def maybe_compress(sess: dict, persona: dict, model: str) -> tuple:
    """
    若到达压缩轮次（turn_count % COMPRESS_EVERY == 0）则压缩本段对话。
    文字（server.py）与语音（voice_server.py）两路共用。
    返回 (memory_updated, profile_updated)。
    """
    if sess["turn_count"] % COMPRESS_EVERY != 0:
        return False, False

    to_compress = sess["messages"][: COMPRESS_EVERY * 2]
    old_summary = sess["memory_summary"]

    result = compress_session(
        to_compress,
        sess["memory_summary"],
        sess["user_profile"],
        persona["display_name"],
        model,
    )

    # 把旧摘要存入记忆链
    if old_summary:
        start = sess["turn_count"] - COMPRESS_EVERY * 2 + 1
        end   = sess["turn_count"] - COMPRESS_EVERY
        sess["memory_chain"].append({
            "label": f"第{max(1,start)}-{end}轮",
            "brief": old_summary[:60] + ("…" if len(old_summary) > 60 else ""),
        })

    old_profile = sess["user_profile"]
    sess["memory_summary"] = result["summary"]
    sess["user_profile"]   = result["user_profile"]
    sess["messages"]       = sess["messages"][COMPRESS_EVERY * 2:]
    print(f"[Memory] turn={sess['turn_count']} 压缩完成 | chain={len(sess['memory_chain'])}")
    return True, sess["user_profile"] != old_profile


# ── 情绪标签 ──────────────────────────────────────────────────────────────────

def extract_emotion(text: str) -> tuple:
    """剥离开头的 <emotion>xxx</emotion>，返回 (emotion, clean_text)。健壮处理畸形标签。"""
    return parse_lead_emotion(text.strip())


# ── Session 管理 ──────────────────────────────────────────────────────────────

sessions: dict = {}   # key → {messages, memory_summary, user_profile, memory_chain, turn_count, persona_id}

def get_session(sid: str, persona_id: str) -> dict:
    key = f"{sid}_{persona_id}"
    if key not in sessions:
        sessions[key] = {
            "messages":       [],
            "memory_summary": "",
            "user_profile":   "",
            "memory_chain":   [],   # [{label, brief}]
            "turn_count":     0,
            "persona_id":     persona_id,
        }
    return sessions[key]
