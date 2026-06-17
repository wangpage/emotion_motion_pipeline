"""
SouyanSkeleton23 —— 本机器人的统一骨架对象。

借鉴 GR00T-WholeBodyControl / MotionBricks 的 G1Skeleton34 思路
(motionbricks/docs/motion_representation.md)：把"关节名 → MuJoCo qpos 索引 → 限位"
固化成一个对象，一举消灭项目里的"三套命名"乱象（语义名 / l*.STL / j*）。

数据来源：urdf/0403_souyan-robot_asm-3-23-21.urdf 的关节限位，
以及 .mujoco_cache_fullmesh_v5/*.fullmesh_v5.xml 的 body 树顺序。
"""
from __future__ import annotations
import math
from dataclasses import dataclass

# 23 个驱动关节，顺序 = fullmesh_v5.xml 的文档顺序 = MuJoCo qpos 顺序（根之后）
JOINT_NAMES = [
    "waist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_pitch_joint",
    "neck_pitch_joint", "neck_yaw_joint", "face_joint",
    "waist_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_knee_joint", "right_ankle_pitch_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint", "left_ankle_pitch_joint",
]
JOINT_COUNT = len(JOINT_NAMES)  # 23
JOINT_INDEX = {n: i for i, n in enumerate(JOINT_NAMES)}

# 关节限位 (lower, upper)（弧度）。continuous 关节用 None 表示无硬限位（做软限幅）。
# 取自 URDF；continuous: 手腕、膝。
LIMITS: dict[str, tuple[float | None, float | None]] = {
    "waist_yaw_joint": (-0.5236, 0.5236),
    "right_shoulder_pitch_joint": (-0.7854, 3.1416),
    "right_shoulder_roll_joint": (-1.2217, 1.2217),
    "right_shoulder_yaw_joint": (-3.1416, 3.1416),
    "right_elbow_joint": (-1.5708, 0.0),
    "right_wrist_pitch_joint": (None, None),     # continuous
    "left_shoulder_pitch_joint": (-0.7854, 3.1416),
    "left_shoulder_roll_joint": (-1.2217, 1.2217),
    "left_shoulder_yaw_joint": (-3.1416, 3.1416),
    "left_elbow_joint": (-1.5708, 0.0),
    "left_wrist_pitch_joint": (None, None),      # continuous
    "neck_pitch_joint": (-0.5236, 0.5236),
    "neck_yaw_joint": (-1.2217, 1.2217),
    "face_joint": (-0.5236, 0.5236),
    "waist_roll_joint": (-0.5236, 0.7854),
    "right_hip_pitch_joint": (-1.0472, 2.0944),
    "right_hip_roll_joint": (-1.2217, 1.2217),
    "right_knee_joint": (None, None),            # continuous
    "right_ankle_pitch_joint": (0.0, 1.7453),
    "left_hip_pitch_joint": (-1.0472, 2.0944),
    "left_hip_roll_joint": (-1.2217, 1.2217),
    "left_knee_joint": (None, None),             # continuous
    "left_ankle_pitch_joint": (0.0, 1.7453),
}

# continuous 关节的软限幅（弧度），避免生成模型给出怪异大角度
SOFT_LIMIT = 1.5708


@dataclass
class ClampReport:
    clamped: int = 0          # 被钳制的 (帧,关节) 次数
    joints: set | None = None  # 触发钳制的关节名集合

    def __post_init__(self):
        if self.joints is None:
            self.joints = set()


class SouyanSkeleton23:
    """统一骨架：关节名 / qpos 索引 / 限位 / 钳制。"""

    joint_names = JOINT_NAMES
    n_joints = JOINT_COUNT

    def __init__(self):
        # qpos 布局：[0:3] 根平移, [3:7] 根四元数(wxyz), [7:7+23] 23 关节
        self.root_qpos_dim = 7
        self.qpos_dim = self.root_qpos_dim + JOINT_COUNT  # 30

    # ---- qpos 索引解析（运行时按名字查，绝不硬编码数字）----
    def resolve_qpos_indices(self, mj_model) -> list[int]:
        """从已加载的 mj_model 按关节名解析每个关节的 qpos 地址。

        这是 MotionBricks 的做法：换模型版本/关节增删也不会错位。
        需要 mujoco 已安装。返回长度 23 的列表，第 i 项 = JOINT_NAMES[i] 的 qpos 索引。
        """
        import mujoco
        idx = []
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"关节 {name} 不在 MuJoCo 模型里")
            idx.append(int(mj_model.jnt_qposadr[jid]))
        return idx

    def static_qpos_indices(self) -> list[int]:
        """不依赖 mujoco 的静态布局（根 7 维之后顺序排列）。
        与 fullmesh_v5.xml 的文档顺序一致；仅当无法加载模型时作回退。"""
        return [self.root_qpos_dim + i for i in range(JOINT_COUNT)]

    # ---- 限位钳制（安全过滤器）----
    def clamp_angle(self, joint: str, value: float) -> float:
        lo, hi = LIMITS[joint]
        if lo is None:  # continuous → 软限幅
            return max(-SOFT_LIMIT, min(SOFT_LIMIT, value))
        return max(lo, min(hi, value))

    def clamp_vector(self, angles, report: ClampReport | None = None):
        """对长度 23 的关节角向量逐关节钳制。原地安全，返回新列表。"""
        out = []
        for i, name in enumerate(JOINT_NAMES):
            v = float(angles[i])
            c = self.clamp_angle(name, v)
            if report is not None and abs(c - v) > 1e-9:
                report.clamped += 1
                report.joints.add(name)
            out.append(c)
        return out

    # ---- 中性姿态 ----
    def neutral(self) -> list[float]:
        """所有关节回 0（ankle 下限为 0，0 合法）。"""
        return [0.0] * JOINT_COUNT


# ---- 6D 旋转表示工具（用于运动流的平滑表示，借鉴 SONIC/MotionBricks）----
# 单轴 hinge 关节其实标量角即可；6D 主要用于人体侧的旋转。这里给最常用的转换。
def angle_to_6d(theta: float) -> list[float]:
    """绕 Z 轴的角 → 6D 连续旋转表示的前两列。"""
    c, s = math.cos(theta), math.sin(theta)
    return [c, -s, 0.0, s, c, 0.0]


def sixd_to_angle(v) -> float:
    """6D 表示 → 绕 Z 轴角（取 atan2，假设近似平面旋转）。"""
    return math.atan2(v[3], v[0])


if __name__ == "__main__":
    sk = SouyanSkeleton23()
    print(f"SouyanSkeleton23: {sk.n_joints} joints, qpos_dim={sk.qpos_dim}")
    rep = ClampReport()
    over = [9.0] * 23  # 全部超限
    sk.clamp_vector(over, rep)
    print(f"clamp 测试: {rep.clamped} 次钳制, 涉及 {len(rep.joints)} 个关节")
