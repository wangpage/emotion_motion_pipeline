"""
Stage 2 —— 动作生成 (Motion Generation)，机器人无关版。

ProceduralGenerator：intent → 情绪库(角色化+范围) → 采样落到该机器人真实关节 → 关键帧编译
→ 叠加二次运动（低频呼吸/微摆，使动作连续自然而非僵在一个状态）→ 限位钳制 → 正运动学坐标。

“范围”体现在两处：①库里每目标 {c,s} 运行时采样（峰值姿态每次不同）；②二次运动叠加持续微动。
两者共同保证：同一情绪每次播放都是一段略不同的、连续自然的轨迹。

mock / mld 后端走人体重定向链路（次要路径，见 retarget.py），未接入时回退 procedural。
"""
from __future__ import annotations

import numpy as np

from pipeline.intent import ActionSpec
from pipeline.motion import RobotMotion, compile_keyframes
from pipeline.library import EmotionLibrary
from core.robot import RobotModel
from core.semantic import SemanticMapper


def add_secondary_motion(body: np.ndarray, fps: float, rng, amp: float = 0.03,
                         taper: float = 0.4, cols=None) -> np.ndarray:
    """给关节叠加低频平滑微动（两正弦叠加），制造“呼吸/微摆”的生命感。

    amp 弧度量级很小，不破坏主动作；首尾各 taper 秒淡入淡出，保证起落平稳。
    cols=None 时作用于全部关节；给定列索引时只作用于这些关节（避免给静止关节加抖动）。
    """
    n, nj = body.shape
    if n < 2:
        return body
    t = np.arange(n) / fps
    env = np.ones(n)
    k = max(1, int(taper * fps))
    if 2 * k < n:
        env[:k] = np.linspace(0, 1, k)
        env[-k:] = np.linspace(1, 0, k)
    for j in (range(nj) if cols is None else cols):
        f1, f2 = rng.uniform(0.15, 0.35), rng.uniform(0.4, 0.7)
        p1, p2 = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
        sig = 0.6 * np.sin(2 * np.pi * f1 * t + p1) + 0.4 * np.sin(2 * np.pi * f2 * t + p2)
        body[:, j] = body[:, j] + amp * env * sig
    return body


class ProceduralGenerator:
    """意图 → 机器人原生关键帧 → 连续自然帧流。机器人无关。"""

    def __init__(self, robot: RobotModel, mapping: dict | None = None,
                 library: EmotionLibrary | None = None, seed: int | None = None,
                 secondary: bool = True):
        self.robot = robot
        if mapping is None:
            mapping, self.report = SemanticMapper().map(robot)
        else:
            self.report = None
        self.mapping = mapping
        self.lib = library or EmotionLibrary.load()
        self.seed = seed
        self.secondary = secondary

    def generate(self, spec: ActionSpec) -> RobotMotion:
        rng = np.random.default_rng(self.seed)
        kfs, fps, used = self.lib.resolve(spec.intent_key, self.mapping,
                                          intensity=spec.intensity, rng=rng)
        m = compile_keyframes(kfs, self.robot.joint_names, fps=fps, name=spec.intent_key)
        if self.secondary:
            jidx = {n: i for i, n in enumerate(self.robot.joint_names)}
            cols = sorted({jidx[jn] for kf in kfs for jn in kf["joints"] if jn in jidx})
            add_secondary_motion(m.body, fps, rng, cols=cols or None)
        m.smooth(window=3)
        _, rep = m.clamp(self.robot)
        m.compute_fk(self.robot)
        m.meta = {
            "backend": "procedural", "intent": spec.intent_key, "intensity": spec.intensity,
            "description": spec.description_en, "clamped": rep.clamped,
            "roles_used": sorted(used), "mapping_source": getattr(self.report, "source", "given"),
        }
        return m


class MLDGenerator:
    """真·一句话生成：description_en → MLD(SMPL) → 重定向 → 机器人。未接入回退 procedural。"""

    def __init__(self, robot: RobotModel, mapping: dict | None = None, mld_fn=None,
                 library: EmotionLibrary | None = None):
        from pipeline.retarget import Retargeter
        self.robot = robot
        self.rt = Retargeter(robot, mapping)
        self.mld_fn = mld_fn
        self._fallback = ProceduralGenerator(robot, mapping, library)

    def generate(self, spec: ActionSpec) -> RobotMotion:
        if self.mld_fn is None:
            print("[generate] 未接入 MLD（mld_fn=None），回退 procedural。见 README『接入 MLD』。")
            return self._fallback.generate(spec)
        smpl_seq = self.mld_fn(spec.description_en)        # (T,22,3)
        m, rep = self.rt.retarget(smpl_seq, name="mld_" + spec.intent_key)
        m.smooth(window=3)
        m.compute_fk(self.robot)
        m.meta = {"backend": "mld", "clamped": rep.clamped, "description": spec.description_en}
        return m


class MockHumanGenerator:
    """合成一段 SMPL-22 动作走重定向链路，离线验证 retarget（无需 MLD）。"""

    def __init__(self, robot: RobotModel, mapping: dict | None = None, **_):
        from pipeline.retarget import Retargeter
        from pipeline.retarget import SMPL_IDX  # noqa: F401
        self.robot = robot
        self.rt = Retargeter(robot, mapping)

    def generate(self, spec: ActionSpec) -> RobotMotion:
        from pipeline.retarget import SMPL_IDX
        T, fps = 60, 30.0
        seq = np.zeros((T, 22, 3))
        ts = np.linspace(0, 2 * np.pi, T)
        seq[:, SMPL_IDX["right_shoulder"], 0] = 1.0 + 0.3 * np.sin(ts)
        seq[:, SMPL_IDX["right_elbow"], 2] = -0.6 + 0.4 * np.sin(2 * ts)
        m, rep = self.rt.retarget(seq, fps=fps, name="mock_" + spec.intent_key)
        m.smooth(window=3)
        m.compute_fk(self.robot)
        m.meta = {"backend": "mock_human", "clamped": rep.clamped}
        return m


def make_generator(backend: str, robot: RobotModel, mapping: dict | None = None,
                   library: EmotionLibrary | None = None, mld_fn=None, seed: int | None = None):
    if backend == "procedural":
        return ProceduralGenerator(robot, mapping, library, seed=seed)
    if backend == "mock":
        return MockHumanGenerator(robot, mapping)
    if backend == "mld":
        return MLDGenerator(robot, mapping, mld_fn=mld_fn, library=library)
    raise ValueError(backend)
