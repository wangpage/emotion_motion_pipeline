"""
语义角色词表 —— 情绪库与具体机器人之间的“中间语言”。

情绪库（config/emotions.json）不写具体关节名（那会绑死某台机器人），改写**语义角色**：
每个角色描述“身体哪个部位、绕哪个解剖自由度、正方向是什么”。SemanticMapper 负责把
任意 URDF 的真实关节映射到这些角色，并给每个角色一个 sign，使“角色值 = 正”在任何机器人
上都表示同一个解剖动作（沿用 README 标定表的语义）。

机体参考系（与 README 一致）：机器人面朝 +X，上为 +Z，左为 +Y（右为 −Y）。

角色命名规范
------------
- 单体角色（脖子/腰/头，机器人通常只有一个）：直接用角色名，如 "neck_pitch"。
- 成对角色（手臂/腿，左右各一）：情绪库写一次（如 "shoulder_pitch"），引擎自动展开到
  左右两侧并各套各自的 sign，得到对称动作。
- 指定单侧（挥手、撒娇这类不对称动作）：加前缀 "R."/"L."，如 "R.shoulder_pitch"。

正方向约定（anatomical-positive，正值 = 括号里的方向）
- neck_pitch       抬头 / 仰
- neck_yaw         头转向左
- head_tilt        头侧倾（face/第三颈自由度）
- waist_yaw        上身左拧
- waist_roll       上身右倾
- waist_pitch      上身前倾
- shoulder_pitch   手臂向前/上摆
- shoulder_abduct  手臂外展张开（远离躯干）
- shoulder_yaw     上臂外旋
- elbow_flex       屈肘（前臂抬起靠近身体）
- wrist_pitch      手腕上抬
- hip_pitch        大腿前抬
- hip_abduct       腿外展
- hip_yaw          腿外旋
- knee_flex        屈膝
- ankle_pitch      脚尖上勾
"""
from __future__ import annotations

SIDES = ("R", "L")

# 单体角色：机器人通常只有一个（脖子/头/腰）
SINGLE_ROLES = (
    "neck_pitch", "neck_yaw", "head_tilt",
    "waist_yaw", "waist_roll", "waist_pitch",
)

# 成对角色：左右各一，情绪库写一次自动展开
PAIRED_ROLES = (
    "shoulder_pitch", "shoulder_abduct", "shoulder_yaw",
    "elbow_flex", "wrist_pitch",
    "hip_pitch", "hip_abduct", "hip_yaw",
    "knee_flex", "ankle_pitch",
)

# 展开后的完整角色键集合：单体 + R./L. 前缀的成对角色
ALL_ROLES = tuple(SINGLE_ROLES) + tuple(
    f"{s}.{r}" for r in PAIRED_ROLES for s in SIDES
)


def is_paired(role: str) -> bool:
    """裸成对角色名（不带 R./L. 前缀）？"""
    return role in PAIRED_ROLES


def expand_role(role: str):
    """把情绪库里的角色键展开成具体角色键列表。

    - "shoulder_pitch"   -> ["R.shoulder_pitch", "L.shoulder_pitch"]（成对自动双侧）
    - "R.shoulder_pitch" -> ["R.shoulder_pitch"]（已指定单侧）
    - "neck_pitch"       -> ["neck_pitch"]（单体）
    未知角色返回空列表（上层据此跳过）。
    """
    if role in SINGLE_ROLES:
        return [role]
    if "." in role:
        side, base = role.split(".", 1)
        if side in SIDES and base in PAIRED_ROLES:
            return [role]
        return []
    if role in PAIRED_ROLES:
        return [f"{s}.{role}" for s in SIDES]
    return []


# 角色 → 解剖自由度（pitch/roll/yaw），供启发式分类与 sign 推断使用
ROLE_DOF = {
    "neck_pitch": "pitch", "neck_yaw": "yaw", "head_tilt": "roll",
    "waist_yaw": "yaw", "waist_roll": "roll", "waist_pitch": "pitch",
    "shoulder_pitch": "pitch", "shoulder_abduct": "roll", "shoulder_yaw": "yaw",
    "elbow_flex": "pitch", "wrist_pitch": "pitch",
    "hip_pitch": "pitch", "hip_abduct": "roll", "hip_yaw": "yaw",
    "knee_flex": "pitch", "ankle_pitch": "pitch",
}

# 身体部位 → 该部位下可能出现的角色（按自由度），供按名+轴归类
PART_ROLES = {
    "neck": ["neck_pitch", "neck_yaw", "head_tilt"],
    "head": ["head_tilt", "neck_pitch", "neck_yaw"],
    "waist": ["waist_yaw", "waist_roll", "waist_pitch"],
    "torso": ["waist_yaw", "waist_roll", "waist_pitch"],
    "spine": ["waist_pitch", "waist_yaw", "waist_roll"],
    "shoulder": ["shoulder_pitch", "shoulder_abduct", "shoulder_yaw"],
    "elbow": ["elbow_flex"],
    "wrist": ["wrist_pitch"],
    "hip": ["hip_pitch", "hip_abduct", "hip_yaw"],
    "knee": ["knee_flex"],
    "ankle": ["ankle_pitch"],
}
