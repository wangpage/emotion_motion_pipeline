"""
SemanticMapper —— 把任意机器人的真实关节自动映射到语义角色。

输出：{role_key: (joint_name, sign)}，role_key 取自 roles.ALL_ROLES（单体如 "neck_pitch"，
成对如 "R.shoulder_pitch"）。sign 使“角色值 = 正”在该机器人上表示 roles.py 里定义的解剖正方向。

两条路径：
1. **内置 profile**（souyan）：用从已在 MuJoCo 标定过的动作库反推出的精确映射 + 符号，命中即用，
   保证已知机器人 100% 正确。
2. **通用启发式**：关节名（部位/侧别/自由度）+ 关节轴在机体系下的方向 + 运动树位置，尽力推断。
   通用路径的符号是 best-effort —— 以仿真/画面验收为准（README 早有提醒，换轴极易搞反）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .roles import SIDES, SINGLE_ROLES, PAIRED_ROLES, ROLE_DOF, PART_ROLES
from .robot import RobotModel, Joint


# ──────────────────────────────────────────────────────────────────────────────
# 1) souyan 内置 profile：role_key -> (joint_name, sign)
#    符号来自 web_viewer/actions.js 的 LIB（已对 MuJoCo 渲染图标定）：
#    - 屈肘 = 关节负值 → elbow_flex sign = -1
#    - 右肩外展 = 关节正、左肩外展 = 关节负 → shoulder_abduct R+ / L-
#    - shoulder_pitch / 单体关节方向与库值一致 → sign +1
# ──────────────────────────────────────────────────────────────────────────────
SOUYAN_PROFILE = {
    "neck_pitch":  ("neck_pitch_joint", +1),
    "neck_yaw":    ("neck_yaw_joint", +1),
    "head_tilt":   ("face_joint", +1),
    "waist_yaw":   ("waist_yaw_joint", +1),
    "waist_roll":  ("waist_roll_joint", +1),

    "R.shoulder_pitch":  ("right_shoulder_pitch_joint", +1),
    "L.shoulder_pitch":  ("left_shoulder_pitch_joint", +1),
    "R.shoulder_abduct": ("right_shoulder_roll_joint", +1),
    "L.shoulder_abduct": ("left_shoulder_roll_joint", -1),
    "R.shoulder_yaw":    ("right_shoulder_yaw_joint", +1),
    "L.shoulder_yaw":    ("left_shoulder_yaw_joint", -1),
    "R.elbow_flex":      ("right_elbow_joint", -1),
    "L.elbow_flex":      ("left_elbow_joint", -1),
    "R.wrist_pitch":     ("right_wrist_pitch_joint", +1),
    "L.wrist_pitch":     ("left_wrist_pitch_joint", -1),

    "R.hip_pitch":   ("right_hip_pitch_joint", +1),
    "L.hip_pitch":   ("left_hip_pitch_joint", +1),
    "R.hip_abduct":  ("right_hip_roll_joint", +1),
    "L.hip_abduct":  ("left_hip_roll_joint", -1),
    "R.knee_flex":   ("right_knee_joint", -1),
    "L.knee_flex":   ("left_knee_joint", -1),
    "R.ankle_pitch": ("right_ankle_pitch_joint", +1),
    "L.ankle_pitch": ("left_ankle_pitch_joint", +1),
}


@dataclass
class MapReport:
    source: str = "heuristic"            # "profile:souyan" / "heuristic"
    mapped: dict = field(default_factory=dict)   # role_key -> joint_name
    unmapped_joints: list = field(default_factory=list)  # 没认出角色的关节名

    def summary(self) -> str:
        return (f"[semantic] 来源={self.source}  识别角色={len(self.mapped)}  "
                f"未识别关节={len(self.unmapped_joints)}"
                + (f"（{', '.join(self.unmapped_joints)}）" if self.unmapped_joints else ""))


# ── 名字解析 ──────────────────────────────────────────────────────────────────
_PART_PAT = re.compile(r"shoulder|elbow|wrist|neck|head|face|waist|torso|spine|hip|knee|ankle", re.I)
_DOF_PAT = re.compile(r"pitch|roll|yaw", re.I)


def _name_part(name: str) -> str | None:
    m = _PART_PAT.search(name)
    if not m:
        return None
    p = m.group(0).lower()
    return {"face": "head", "torso": "waist", "spine": "waist"}.get(p, p)


def _name_side(name: str) -> str | None:
    n = name.lower()
    if re.search(r"(^|[^a-z])(right|_r_|_r$|\br\b)", n) or n.startswith("r_") or "right" in n:
        return "R"
    if re.search(r"(^|[^a-z])(left|_l_|_l$|\bl\b)", n) or n.startswith("l_") or "left" in n:
        return "L"
    return None


def _name_dof(name: str) -> str | None:
    m = _DOF_PAT.search(name)
    return m.group(0).lower() if m else None


# 机体系：X 前, Y 左, Z 上。轴主分量 → 自由度。
# rotation about Y → pitch(点头/抬臂)；about X → roll(侧倾/外展)；about Z → yaw(转)
_AXIS_DOF = {0: "roll", 1: "pitch", 2: "yaw"}
# 通用解剖正方向常量（标准人形；souyan 等非标走 profile）
_ANATO = {"pitch": +1, "roll": +1, "yaw": +1}


class SemanticMapper:
    def __init__(self, soft_profiles: bool = True):
        self.soft_profiles = soft_profiles

    # ---- 入口 ----
    def map(self, robot: RobotModel):
        prof = self._match_profile(robot)
        if prof is not None:
            mapping, src = prof
        else:
            mapping = self._heuristic(robot)
            src = "heuristic"
        used = {jn for jn, _ in mapping.values()}
        report = MapReport(
            source=src,
            mapped={k: v[0] for k, v in mapping.items()},
            unmapped_joints=[n for n in robot.joint_names if n not in used],
        )
        return mapping, report

    # ---- 内置 profile 匹配 ----
    def _match_profile(self, robot: RobotModel):
        names = set(robot.joint_names)
        sig = {"right_shoulder_pitch_joint", "neck_pitch_joint", "waist_yaw_joint"}
        if sig.issubset(names):
            mapping = {rk: (jn, sg) for rk, (jn, sg) in SOUYAN_PROFILE.items() if jn in names}
            return mapping, "profile:souyan"
        return None

    # ---- 通用启发式 ----
    def _heuristic(self, robot: RobotModel):
        frames = robot.frames()
        pos = robot.fk({})
        # 候选：(joint, part, side, dof, axis_body)
        cands = []
        for j in robot.movable:
            part = _name_part(j.name)
            if part is None:
                continue
            ab = robot.axis_in_body(j, frames)
            dof = _name_dof(j.name) or _AXIS_DOF[int(np.argmax(np.abs(ab)))]
            side = _name_side(j.name)
            if side is None:
                cy = pos.get(j.child, np.zeros(3))[1]
                side = "R" if cy < -1e-3 else ("L" if cy > 1e-3 else None)
            cands.append((j, part, side, dof, ab))

        mapping: dict[str, tuple[str, int]] = {}
        for j, part, side, dof, ab in cands:
            role = self._role_for(part, dof)
            if role is None:
                continue
            paired = role in PAIRED_ROLES
            key = f"{side}.{role}" if paired else role
            if paired and side is None:
                continue
            if key in mapping:               # 同角色多关节：保留先出现（更靠近躯干）
                continue
            mapping[key] = (j.name, self._sign(role, dof, side, ab, j))
        return mapping

    @staticmethod
    def _role_for(part: str, dof: str) -> str | None:
        roles = PART_ROLES.get(part, [])
        # 优先 dof 对得上的角色
        for r in roles:
            if ROLE_DOF.get(r) == dof:
                return r
        return roles[0] if roles else None

    @staticmethod
    def _sign(role: str, dof: str, side: str | None, ab: np.ndarray, joint: Joint = None) -> int:
        """best-effort 符号。

        - elbow/knee 屈曲：按限位可用方向定号——朝 ROM 更大的一侧屈（解剖正=屈）。
          这样无论 URDF 把屈曲定为正或负都能弯起来（souyan 肘 (-1.57,0)→负；通用 (0,2.5)→正）。
        - 其余：归一化到“+值=绕 +机体轴”，叠加左右镜像。
        """
        if role in ("elbow_flex", "knee_flex") and joint is not None \
                and joint.lower is not None and joint.upper is not None:
            return 1 if abs(joint.upper) >= abs(joint.lower) else -1
        di = {"roll": 0, "pitch": 1, "yaw": 2}[dof]
        base = 1 if ab[di] >= 0 else -1
        mirror = -1 if (side == "L" and dof in ("roll", "yaw")) else 1
        flex_bias = -1 if role in ("elbow_flex", "knee_flex") else 1
        return base * _ANATO.get(dof, 1) * mirror * flex_bias


if __name__ == "__main__":
    rm = RobotModel.from_file("assets/souyan.urdf")
    mp, rep = SemanticMapper().map(rm)
    print(rm)
    print(rep.summary())
    for k in sorted(mp):
        print(f"  {k:20s} -> {mp[k][0]:28s} sign={mp[k][1]:+d}")
