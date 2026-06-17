"""
Stage 1 —— 意图放大 (Intention Amplification)。

把"非常委屈地揉眼睛"这种情绪/口语，翻译成结构化的、动作生成模型听得懂的英文描述。
PRD §3：直接把情绪词喂给 MLD 会懵，先过一层 LLM 翻译。

两个后端：
- RuleBasedAmplifier：零依赖、零成本，关键词 → 标准意图 + 强度副词。今天就能跑。
- LLMAmplifier：调 Claude / OpenAI / 本地 Qwen，质量更好。设 env 即启用，否则自动回退到规则。
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass


@dataclass
class ActionSpec:
    """意图放大的产物，供下游生成器使用。"""
    raw_text: str                 # 原始输入
    emotion: str                  # 6 情绪之一 或 'intent'
    intent_key: str               # 规范意图键（procedural 生成器据此查动作）
    description_en: str           # 给 MLD 的英文动作描述
    intensity: float = 1.0        # 0~1.5，强度（缩放动作幅度）


# 6 情绪 + 意图 的规范英文描述模板（PRD §3 表）
TEMPLATES = {
    "happy":   "A person raises both arms outward and nods the head with the upper body leaning slightly forward.",
    "sad":     "A person lowers the head, draws both shoulders inward, and shrinks the upper body slightly.",
    "angry":   "A person leans the torso forward, presses both arms downward and tense, with the head slightly lowered.",
    "surprise":"A person quickly raises the head, opens both arms wide, and leans the upper body backward.",
    "fear":    "A person turns the head aside and pulls back, raising both arms to guard in front of the chest.",
    "neutral": "A person stands still and breathes calmly with tiny idle motions of the head.",
    # 具体意图
    "grieved": "A person lowers the head slightly, brings both hands up to the face, and moves the wrists left and right near the eyes.",
    "wave":    "A person raises the right arm and waves the hand side to side to say hello.",
    "nod":     "A person nods the head up and down.",
    "shake":   "A person shakes the head left and right to say no.",
    "coy":     "A person tilts the head, raises one hand near the face, and sways the upper body shyly.",
    "shy":     "A person lowers and turns the head away while raising both forearms toward the face.",
}

# 中文/口语关键词 → 规范意图键
KEYWORDS = [
    (("委屈", "揉眼", "抹眼", "哭"), "grieved"),
    (("开心", "高兴", "快乐", "太好了", "棒"), "happy"),
    (("难过", "伤心", "沮丧", "失落"), "sad"),
    (("生气", "愤怒", "气", "火大"), "angry"),
    (("惊讶", "吃惊", "震惊", "哇"), "surprise"),
    (("害怕", "恐惧", "怕", "吓", "担心"), "fear"),
    (("撒娇",), "coy"),
    (("害羞", "羞"), "shy"),
    (("打招呼", "挥手", "你好", "hi", "hello", "再见", "拜拜"), "wave"),
    (("点头", "同意", "好的"), "nod"),
    (("摇头", "不行", "不要"), "shake"),
    (("待机", "中性", "平静", "站", "idle"), "neutral"),
]

# 强度副词
STRONG = ("非常", "特别", "超级", "极其", "好", "很")
WEAK = ("有点", "稍微", "略", "一点")

EMOTION_OF = {  # intent_key → 归属情绪（用于上层统计/缓存）
    "grieved": "sad", "wave": "intent", "nod": "intent", "shake": "intent",
    "coy": "happy", "shy": "fear",
    **{k: k for k in ("happy", "sad", "angry", "surprise", "fear", "neutral")},
}


def _parse_intensity(text: str) -> float:
    if any(w in text for w in STRONG):
        return 1.3
    if any(w in text for w in WEAK):
        return 0.7
    return 1.0


class RuleBasedAmplifier:
    """关键词规则版意图放大。零依赖。"""

    def amplify(self, text: str) -> ActionSpec:
        t = text.strip().lower()
        intent_key = "neutral"
        for kws, key in KEYWORDS:
            if any(k.lower() in t for k in kws):
                intent_key = key
                break
        intensity = _parse_intensity(text)
        return ActionSpec(
            raw_text=text,
            emotion=EMOTION_OF.get(intent_key, "intent"),
            intent_key=intent_key,
            description_en=TEMPLATES[intent_key],
            intensity=intensity,
        )


class LLMAmplifier:
    """LLM 版意图放大。需要 ANTHROPIC_API_KEY（或改 provider）。失败自动回退规则版。"""

    SYSTEM = (
        "You translate a short Chinese emotion/utterance into a concise English body-motion "
        "description for a humanoid motion-generation model. Output ONLY the description, one "
        "sentence, structured as: 'A person <body part> <action> <direction/manner>'. "
        "Focus on head, arms, torso. The robot has no fingers."
    )

    def __init__(self, model: str = "claude-opus-4-8"):
        self.model = model
        self._fallback = RuleBasedAmplifier()

    def amplify(self, text: str) -> ActionSpec:
        try:
            import anthropic  # 延迟导入
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=self.model, max_tokens=120, system=self.SYSTEM,
                messages=[{"role": "user", "content": text}],
            )
            desc = msg.content[0].text.strip()
        except Exception as e:  # 无 key / 无网络 / 无包 → 回退
            spec = self._fallback.amplify(text)
            spec.meta = getattr(spec, "meta", {})
            print(f"[intent] LLM 不可用，回退规则版 ({type(e).__name__})")
            return spec
        # 用规则版定 intent_key/emotion/intensity，用 LLM 文本作描述（最佳混合）
        base = self._fallback.amplify(text)
        base.description_en = desc
        return base


def make_amplifier(backend: str = "auto"):
    """backend: 'rule' | 'llm' | 'auto'(有 ANTHROPIC_API_KEY 用 llm 否则 rule)"""
    if backend == "rule":
        return RuleBasedAmplifier()
    if backend == "llm":
        return LLMAmplifier()
    if backend == "auto":
        return LLMAmplifier() if os.getenv("ANTHROPIC_API_KEY") else RuleBasedAmplifier()
    raise ValueError(backend)


if __name__ == "__main__":
    amp = RuleBasedAmplifier()
    for s in ["非常委屈地揉眼睛", "我好开心啊", "你好呀挥挥手", "有点害怕"]:
        spec = amp.amplify(s)
        print(f"{s:16s} -> {spec.intent_key:8s} x{spec.intensity}  | {spec.description_en}")
