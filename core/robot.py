"""
RobotModel —— 解析任意 URDF，提供关节信息 / 限位钳制 / 最小正运动学。

取代旧的 souyan 专用 SouyanSkeleton23：关节数、关节名、限位、轴全部从上传的 URDF 读出，
不再写死。情绪库通过 SemanticMapper 把动作落到这里解析出来的真实关节上。
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

# continuous 关节无硬限位，用软限幅避免怪异大角度（弧度）
SOFT_LIMIT = 1.5708


@dataclass
class Joint:
    name: str
    jtype: str                      # revolute / continuous / prismatic / fixed
    axis: np.ndarray                # (3,) 单位轴（关节坐标系内）
    lower: float | None
    upper: float | None
    parent: str                     # 父 link 名
    child: str                      # 子 link 名
    origin_xyz: np.ndarray          # (3,) 相对父 link 的平移
    origin_rpy: np.ndarray          # (3,) 相对父 link 的 rpy

    @property
    def movable(self) -> bool:
        return self.jtype in ("revolute", "continuous", "prismatic")

    @property
    def continuous(self) -> bool:
        return self.jtype == "continuous"


@dataclass
class ClampReport:
    clamped: int = 0
    joints: set = field(default_factory=set)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _axis_angle_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues：绕单位轴 axis 转 theta 的旋转矩阵。"""
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.eye(3)
    x, y, z = axis / n
    c, s = math.cos(theta), math.sin(theta)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


class RobotModel:
    """从 URDF 文本/文件构建的机器人模型。"""

    def __init__(self, joints: list[Joint], links: list[str], root_link: str,
                 name: str = "robot"):
        self.name = name
        self.joints = joints
        self.links = links
        self.root_link = root_link
        self.joint_by_name = {j.name: j for j in joints}
        self.movable = [j for j in joints if j.movable]
        self.joint_names = [j.name for j in self.movable]   # 动作向量的列顺序
        self.n_joints = len(self.movable)
        # link -> 以它为父的关节列表（建 FK 用）
        self._children: dict[str, list[Joint]] = {}
        for j in joints:
            self._children.setdefault(j.parent, []).append(j)

    # ---------- 解析 ----------
    @classmethod
    def from_string(cls, xml_text: str, name: str = "robot") -> "RobotModel":
        root = ET.fromstring(xml_text)
        return cls._from_root(root, name)

    @classmethod
    def from_file(cls, path: str) -> "RobotModel":
        tree = ET.parse(path)
        root = tree.getroot()
        nm = root.get("name") or path.split("/")[-1]
        return cls._from_root(root, nm)

    @staticmethod
    def _vec(text: str | None, default) -> np.ndarray:
        if not text:
            return np.array(default, dtype=float)
        return np.array([float(x) for x in text.split()], dtype=float)

    @classmethod
    def _from_root(cls, root: ET.Element, name: str) -> "RobotModel":
        links = [l.get("name") for l in root.findall("link") if l.get("name")]
        joints: list[Joint] = []
        for j in root.findall("joint"):
            jt = j.get("type", "fixed")
            ax_el = j.find("axis")
            axis = cls._vec(ax_el.get("xyz") if ax_el is not None else None, [1, 0, 0])
            lim = j.find("limit")
            lo = hi = None
            if lim is not None:
                lo = float(lim.get("lower")) if lim.get("lower") is not None else None
                hi = float(lim.get("upper")) if lim.get("upper") is not None else None
            org = j.find("origin")
            oxyz = cls._vec(org.get("xyz") if org is not None else None, [0, 0, 0])
            orpy = cls._vec(org.get("rpy") if org is not None else None, [0, 0, 0])
            par = j.find("parent"); ch = j.find("child")
            joints.append(Joint(
                name=j.get("name", f"joint{len(joints)}"), jtype=jt, axis=axis,
                lower=lo, upper=hi,
                parent=par.get("link") if par is not None else "",
                child=ch.get("link") if ch is not None else "",
                origin_xyz=oxyz, origin_rpy=orpy,
            ))
        # 根 link = 从不作为任何关节子节点的 link
        child_links = {j.child for j in joints}
        roots = [l for l in links if l not in child_links]
        root_link = roots[0] if roots else (links[0] if links else "base")
        return cls(joints, links, root_link, name=name)

    # ---------- 限位钳制 ----------
    def clamp_value(self, joint_name: str, v: float) -> float:
        j = self.joint_by_name.get(joint_name)
        if j is None:
            return v
        if j.continuous or (j.lower is None or j.upper is None):
            return max(-SOFT_LIMIT, min(SOFT_LIMIT, v))
        return max(j.lower, min(j.upper, v))

    def clamp_vector(self, angles, report: ClampReport | None = None) -> list[float]:
        """对长度 n_joints 的向量（顺序 = self.joint_names）逐关节钳制。"""
        out = []
        for i, name in enumerate(self.joint_names):
            v = float(angles[i])
            c = self.clamp_value(name, v)
            if report is not None and abs(c - v) > 1e-9:
                report.clamped += 1
                report.joints.add(name)
            out.append(c)
        return out

    def neutral(self) -> list[float]:
        """所有关节回 0，再钳到合法区间（有些关节 0 不在限位内）。"""
        return self.clamp_vector([0.0] * self.n_joints)

    # ---------- 最小正运动学 ----------
    def fk(self, angles_by_name: dict[str, float]) -> dict[str, np.ndarray]:
        """给定关节角（按名），返回每个 link 原点的世界坐标 {link: (3,)}。

        根 link 置于世界原点。深度优先遍历运动树累乘变换。
        """
        out: dict[str, np.ndarray] = {self.root_link: np.zeros(3)}
        stack = [(self.root_link, np.eye(4))]
        while stack:
            link, M = stack.pop()
            out[link] = M[:3, 3].copy()
            for j in self._children.get(link, []):
                Torg = _T(_rpy_matrix(j.origin_rpy), j.origin_xyz)
                if j.jtype in ("revolute", "continuous"):
                    Tmot = _T(_axis_angle_matrix(j.axis, angles_by_name.get(j.name, 0.0)),
                              np.zeros(3))
                elif j.jtype == "prismatic":
                    Tmot = _T(np.eye(3), j.axis * angles_by_name.get(j.name, 0.0))
                else:
                    Tmot = np.eye(4)
                stack.append((j.child, M @ Torg @ Tmot))
        return out

    def fk_frame(self, angles) -> np.ndarray:
        """一帧关节向量 → 所有 link 原点世界坐标 (n_links, 3)，顺序 = self.links。"""
        by = {n: float(angles[i]) for i, n in enumerate(self.joint_names)}
        pos = self.fk(by)
        return np.array([pos.get(l, np.zeros(3)) for l in self.links])

    def frames(self, angles_by_name: dict[str, float] | None = None) -> dict[str, np.ndarray]:
        """给定关节角，返回每个 link 的世界 4x4 变换 {link: M}（含旋转）。

        语义映射需要把关节轴从自身坐标系变换到机体系，故需要父 link 的旋转。
        angles_by_name=None 表示中性位（全 0）。
        """
        a = angles_by_name or {}
        out: dict[str, np.ndarray] = {self.root_link: np.eye(4)}
        stack = [(self.root_link, np.eye(4))]
        while stack:
            link, M = stack.pop()
            out[link] = M
            for j in self._children.get(link, []):
                Torg = _T(_rpy_matrix(j.origin_rpy), j.origin_xyz)
                if j.jtype in ("revolute", "continuous"):
                    Tmot = _T(_axis_angle_matrix(j.axis, a.get(j.name, 0.0)), np.zeros(3))
                elif j.jtype == "prismatic":
                    Tmot = _T(np.eye(3), j.axis * a.get(j.name, 0.0))
                else:
                    Tmot = np.eye(4)
                stack.append((j.child, M @ Torg @ Tmot))
        return out

    def axis_in_body(self, joint: "Joint", frames: dict[str, np.ndarray]) -> np.ndarray:
        """关节轴在机体（根）坐标系下的方向（单位向量）。"""
        Mp = frames.get(joint.parent, np.eye(4))
        v = Mp[:3, :3] @ joint.axis
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def __repr__(self):
        return (f"RobotModel({self.name!r}: {self.n_joints} movable joints, "
                f"{len(self.links)} links, root={self.root_link!r})")
