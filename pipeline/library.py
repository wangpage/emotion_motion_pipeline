"""
EmotionLibrary —— 读 config/emotions.json，把「角色化 + 带范围」的情绪关键帧解析成
某台具体机器人的「关节名 + 具体角度」关键帧。

范围怎么落地（这是“动作是一个范围而不是固定值”的实现）：
每个目标 {c, s} 在运行时采样为 c + s·U(-1,1)，所以同一情绪每次播放的峰值姿态都略有不同 →
连续轨迹也随之不同 → 自然、不僵硬。强度 intensity 缩放目标幅度。
"""
from __future__ import annotations

import json
import os

import numpy as np

from core.roles import expand_role

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LIB = os.path.join(_HERE, "config", "emotions.json")


def _sample_t(t, rng) -> float:
    if isinstance(t, (list, tuple)):
        return float(rng.uniform(t[0], t[1]))
    return float(t)


def _sample_val(spec, rng) -> float:
    """{c,s} 或裸数 → 在 [c-s, c+s] 内采样。"""
    if isinstance(spec, dict):
        c = float(spec.get("c", 0.0))
        s = float(spec.get("s", 0.0))
        return c + s * float(rng.uniform(-1.0, 1.0))
    return float(spec)


class EmotionLibrary:
    def __init__(self, data: dict):
        self.data = data
        self.fps = float(data.get("fps", 30.0))
        self.emotions = data.get("emotions", {})
        self.emotion_to_intent = data.get("emotion_to_intent", {})

    @classmethod
    def load(cls, path: str | None = None) -> "EmotionLibrary":
        with open(path or DEFAULT_LIB, encoding="utf-8") as f:
            return cls(json.load(f))

    def has(self, key: str) -> bool:
        return key in self.emotions

    def keys(self):
        return list(self.emotions.keys())

    def intent_for_emotion(self, emotion_zh: str):
        """AIyizhizai 中文情绪词 → (intent_key, intensity)。认不出回退 neutral。"""
        return tuple(self.emotion_to_intent.get(emotion_zh, ["neutral", 1.0]))

    def resolve(self, intent_key: str, mapping: dict, intensity: float = 1.0, rng=None):
        """情绪 → 某机器人具体关键帧。

        mapping: {role_key: (joint_name, sign)}（来自 SemanticMapper）。
        返回 (keyframes, fps, used_roles)：keyframes=[{t, joints:{关节名:角度}, easing}]。
        对不上的角色（机器人没有对应关节）自动跳过。
        """
        rng = rng or np.random.default_rng()
        emo = self.emotions.get(intent_key) or self.emotions.get("neutral", {"keyframes": []})
        fps = float(emo.get("fps", self.fps))
        used_roles = set()

        out = []
        prev_t = -1e9
        for kf in emo.get("keyframes", []):
            t = _sample_t(kf.get("t", 0.0), rng)
            t = max(t, prev_t + 1.0 / fps)   # 保单调，避免采样后关键帧错序
            prev_t = t
            joints: dict[str, float] = {}
            for role, spec in kf.get("pose", {}).items():
                val = _sample_val(spec, rng) * intensity
                for rk in expand_role(role):
                    js = mapping.get(rk)
                    if js is None:
                        continue
                    jname, sign = js
                    joints[jname] = joints.get(jname, 0.0) + sign * val
                    used_roles.add(rk)
            out.append({"t": t, "joints": joints, "easing": kf.get("ez", "ease_in_out")})
        return out, fps, used_roles
