"""
Stage 3 —— 重定向 (Retargeting)：SMPL-22 人体动作 → 任意机器人关节 + 限位钳制。

机器人无关版：不再写死 souyan 关节，而是经「语义角色」桥接——
SMPL 关节 → 角色（ROLE_FROM_SMPL）→ SemanticMapper 给出的该机器人真实关节 + sign。
属次要路径（主路径是 generate.py 的 procedural），供 mock/mld 用。

人体骨架按 HumanML3D / SMPL-22 关节顺序。
"""
from __future__ import annotations

import numpy as np

from core.robot import RobotModel, ClampReport
from core.semantic import SemanticMapper
from core.roles import expand_role

SMPL22 = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]
SMPL_IDX = {n: i for i, n in enumerate(SMPL22)}
AXIS_COL = {"x": 0, "y": 1, "z": 2}

# 角色（裸名/带侧前缀）→ (SMPL 关节基名, 取 axis-angle 的哪个轴分量)
# 成对角色用基名，retarget 时按 R./L. 拼出 right_/left_ 前缀。
ROLE_FROM_SMPL = {
    "neck_pitch": ("neck", "x"), "neck_yaw": ("neck", "y"), "head_tilt": ("head", "x"),
    "waist_yaw": ("spine2", "y"), "waist_roll": ("spine1", "x"), "waist_pitch": ("spine1", "z"),
    "shoulder_pitch": ("shoulder", "x"), "shoulder_abduct": ("shoulder", "z"),
    "shoulder_yaw": ("shoulder", "y"), "elbow_flex": ("elbow", "z"), "wrist_pitch": ("wrist", "x"),
    "hip_pitch": ("hip", "x"), "hip_abduct": ("hip", "z"), "hip_yaw": ("hip", "y"),
    "knee_flex": ("knee", "x"), "ankle_pitch": ("ankle", "x"),
}
_SIDE_PREFIX = {"R": "right_", "L": "left_"}


class Retargeter:
    """SMPL-22 axis-angle 序列 → 机器人 RobotMotion。"""

    def __init__(self, robot: RobotModel, mapping: dict | None = None):
        self.robot = robot
        if mapping is None:
            mapping, _ = SemanticMapper().map(robot)
        self.mapping = mapping
        self.jidx = {n: i for i, n in enumerate(robot.joint_names)}

    def _smpl_name(self, role_key: str, base_role: str) -> str:
        if "." in role_key:
            side = role_key.split(".", 1)[0]
            return _SIDE_PREFIX.get(side, "") + base_role
        return base_role

    def retarget_pose(self, aa: np.ndarray) -> np.ndarray:
        out = np.zeros(self.robot.n_joints)
        for role_key, (jname, sign) in self.mapping.items():
            base = role_key.split(".", 1)[1] if "." in role_key else role_key
            src = ROLE_FROM_SMPL.get(base)
            if src is None or jname not in self.jidx:
                continue
            smpl_base, axis = src
            sname = self._smpl_name(role_key, smpl_base)
            si = SMPL_IDX.get(sname)
            if si is None:
                continue
            out[self.jidx[jname]] = sign * aa[si][AXIS_COL[axis]]
        return out

    def retarget(self, smpl_seq: np.ndarray, fps: float = 30.0, name: str = "retargeted"):
        from pipeline.motion import RobotMotion
        smpl_seq = np.asarray(smpl_seq, dtype=np.float64).reshape(-1, 22, 3)
        body = np.stack([self.retarget_pose(smpl_seq[t]) for t in range(len(smpl_seq))])
        m = RobotMotion(body=body, joint_names=self.robot.joint_names, fps=fps, name=name)
        _, rep = m.clamp(self.robot)
        return m, rep
