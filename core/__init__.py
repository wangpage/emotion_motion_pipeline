"""core —— 与具体机器人解耦的统一引擎。

- roles    : 语义角色词表（情绪库用角色描述动作，而非具体关节名）。
- robot    : RobotModel，解析任意 URDF（关节/轴/限位/树）+ 限位钳制 + 最小正运动学。
- semantic : SemanticMapper，把任意机器人的真实关节自动映射到语义角色 + 方向符号。
"""
from .roles import (  # noqa: F401
    SIDES, SINGLE_ROLES, PAIRED_ROLES, ALL_ROLES, expand_role, is_paired,
)
from .robot import RobotModel  # noqa: F401
from .semantic import SemanticMapper, MapReport  # noqa: F401
