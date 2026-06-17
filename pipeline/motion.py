"""
RobotMotion —— 与具体机器人解耦的统一运动格式（逐帧关节向量 + 可选 FK 坐标）。

取代旧 frames.py（那版 joint 数写死 23、依赖 SouyanSkeleton23）。现在 joint_names 由机器人传入，
关节数任意；新增 positions(T, n_links, 3) 存正运动学算出的每连杆世界坐标。

JSON 同时给出「关节角度轨迹」与「FK 3D 坐标」（满足两种输出诉求）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

DEFAULT_ROOT = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

EASINGS = {
    "linear": lambda t: t,
    "ease_in": lambda t: t * t,
    "ease_out": lambda t: 1 - (1 - t) ** 2,
    "ease_in_out": lambda t: 3 * t * t - 2 * t * t * t,
}


@dataclass
class RobotMotion:
    body: np.ndarray                       # (T, n_joints) 关节角（弧度）
    joint_names: list                      # 长度 n_joints，body 的列含义
    fps: float = 30.0
    root: np.ndarray | None = None         # (T, 7) 根姿态；None=全程站立
    positions: np.ndarray | None = None    # (T, n_links, 3) FK 世界坐标（可选）
    link_names: list | None = None         # positions 的列含义
    name: str = "motion"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.joint_names)
        self.body = np.asarray(self.body, dtype=np.float64).reshape(-1, n)
        if self.root is None:
            self.root = np.tile(DEFAULT_ROOT, (len(self.body), 1))
        else:
            self.root = np.asarray(self.root, dtype=np.float64).reshape(-1, 7)

    @property
    def n_frames(self) -> int:
        return len(self.body)

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    # ---- 后处理 ----
    def clamp(self, robot):
        """逐帧按机器人真实限位钳制。返回 (self, ClampReport)。"""
        from core.robot import ClampReport
        rep = ClampReport()
        for i in range(self.n_frames):
            self.body[i] = robot.clamp_vector(self.body[i], rep)
        return self, rep

    def smooth(self, window: int = 3):
        if window < 2 or self.n_frames < window:
            return self
        k = np.ones(window) / window
        for j in range(self.n_joints):
            self.body[:, j] = np.convolve(self.body[:, j], k, mode="same")
        return self

    def compute_fk(self, robot):
        """用机器人 FK 填充 positions(T, n_links, 3)。"""
        self.link_names = list(robot.links)
        self.positions = np.stack([robot.fk_frame(self.body[i]) for i in range(self.n_frames)])
        return self

    def qpos_frame(self, i: int) -> np.ndarray:
        """第 i 帧 = 根 7 维 + n_joints 关节角（注入仍按名解析散布）。"""
        return np.concatenate([self.root[i], self.body[i]])

    # ---- 序列化 ----
    def to_json(self, path: str):
        obj = {
            "name": self.name, "fps": self.fps, "joint_names": list(self.joint_names),
            "body": self.body.tolist(), "root": self.root.tolist(),
            "meta": self.meta,
        }
        if self.positions is not None:
            obj["link_names"] = list(self.link_names or [])
            obj["positions"] = self.positions.tolist()
        with open(path, "w") as f:
            json.dump(obj, f)

    @classmethod
    def from_json(cls, path: str) -> "RobotMotion":
        with open(path) as f:
            o = json.load(f)
        return cls(
            body=np.array(o["body"]), joint_names=o["joint_names"], fps=o.get("fps", 30.0),
            root=np.array(o["root"]) if o.get("root") else None,
            positions=np.array(o["positions"]) if o.get("positions") else None,
            link_names=o.get("link_names"), name=o.get("name", "motion"), meta=o.get("meta", {}),
        )


def compile_keyframes(keyframes: list, joint_names: list, fps: float = 30.0,
                      name: str = "motion") -> RobotMotion:
    """把已解析（关节名 + 具体角度值）的关键帧编译成逐帧 RobotMotion。

    keyframes: [{"t": 秒, "joints": {关节名: 角度}, "easing": "ease_out"}, ...]
    未列出的关节维持上一关键帧的值；缺省从全 0 开始。沿用旧 frames.py 的缓动插值。
    """
    assert keyframes, "至少要一个关键帧"
    idx = {n: i for i, n in enumerate(joint_names)}
    nj = len(joint_names)
    keyframes = sorted(keyframes, key=lambda k: k["t"])
    total_t = keyframes[-1]["t"]
    n = max(2, int(round(total_t * fps)) + 1)
    body = np.zeros((n, nj))

    full = []
    prev = np.zeros(nj)
    for kf in keyframes:
        vec = prev.copy()
        for jn, val in kf.get("joints", {}).items():
            if jn in idx:
                vec[idx[jn]] = float(val)
        full.append((kf["t"], vec, kf.get("easing", "ease_in_out")))
        prev = vec

    for i in range(n):
        ti = i / fps
        seg = 0
        while seg < len(full) - 1 and ti > full[seg + 1][0]:
            seg += 1
        t0, v0, _ = full[seg]
        t1, v1, ez = full[min(seg + 1, len(full) - 1)]
        if t1 <= t0:
            body[i] = v1
        else:
            a = EASINGS.get(ez, EASINGS["ease_in_out"])((ti - t0) / (t1 - t0))
            body[i] = (1 - a) * v0 + a * v1
    return RobotMotion(body=body, joint_names=list(joint_names), fps=fps, name=name)
