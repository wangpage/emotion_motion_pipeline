"""
Stage 4 —— MuJoCo 注入与播放。

照搬 MotionBricks 验证过的循环 (interactive_demo_g1.py:74-97)：
    mj_data.qpos[:] = qpos ; mujoco.mj_forward(...) ; viewer.sync() ; sleep
本期走 A 路线 = qpos 运动学回放（fullmesh_v5 无 actuator、漂浮基座，qpos 回放最稳）。

mujoco 是可选依赖：
- 装了 mujoco → 真·仿真播放（viewer 可选）。
- 没装 → headless 干跑（逐帧迭代 + 统计），管线照样验证。
"""
from __future__ import annotations
import os
import time
import numpy as np

from core.robot import RobotModel
from pipeline.motion import RobotMotion

# 默认模型路径：项目里的 fullmesh_v5（souyan；meshdir 从该 xml 所在目录解析）
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_HERE)
DEFAULT_MJCF = os.path.join(
    _PROJECT_ROOT,
    ".mujoco_cache_fullmesh_v5",
    "0403_souyan-robot_asm-3-23-21.fullmesh_v5.xml",
)


class MujocoPlayer:
    def __init__(self, robot: RobotModel, mjcf_path: str | None = None,
                 has_floating_base: bool = True):
        self.robot = robot
        self.mjcf_path = mjcf_path or DEFAULT_MJCF
        self.has_floating_base = has_floating_base
        self._mj = None
        self.model = None
        self.data = None
        self.qpos_idx = None  # 各关节在 qpos 里的地址（按名解析）

    # ---- 懒加载 mujoco ----
    def _ensure_loaded(self) -> bool:
        if self.model is not None:
            return True
        try:
            import mujoco
        except ImportError:
            print("[player] 未安装 mujoco，进入 headless 干跑模式。"
                  "\n          安装：pip install mujoco   然后即可看到仿真。")
            return False
        if not os.path.exists(self.mjcf_path):
            print(f"[player] 找不到模型文件：{self.mjcf_path}（headless 干跑）")
            return False
        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)
        # 按关节名解析 qpos 地址（换模型也不会错位）；模型里没有的关节记为 -1 跳过
        idx = []
        for name in self.robot.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            idx.append(int(self.model.jnt_qposadr[jid]) if jid >= 0 else -1)
        self.qpos_idx = idx
        miss = sum(1 for a in idx if a < 0)
        print(f"[player] 已加载模型，nq={self.model.nq}，{len(idx)-miss}/{len(idx)} 关节按名解析"
              + ("" if not miss else f"（{miss} 个不在模型里，已跳过）"))
        return True

    # ---- 写一帧 qpos ----
    def _apply_frame(self, motion: RobotMotion, i: int):
        if self.has_floating_base and self.model.nq >= 7:
            self.data.qpos[0:7] = motion.root[i]
        for j, adr in enumerate(self.qpos_idx):
            if adr >= 0:
                self.data.qpos[adr] = motion.body[i, j]

    # ---- 播放 ----
    def play(self, motion: RobotMotion, viewer: bool = True, loops: int = 1,
             record_dir: str | None = None):
        if not self._ensure_loaded():
            return self._headless(motion, loops)

        mujoco = self._mj
        dt = 1.0 / motion.fps
        renderer = None
        frames_png = []
        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
            renderer = mujoco.Renderer(self.model, height=480, width=640)

        def run_once(sync=None):
            for i in range(motion.n_frames):
                t0 = time.time()
                self._apply_frame(motion, i)
                mujoco.mj_forward(self.model, self.data)   # 运动学更新（A 路线）
                if sync:
                    sync()
                if renderer is not None:
                    renderer.update_scene(self.data)
                    frames_png.append(renderer.render().copy())
                rest = dt - (time.time() - t0)
                if sync and rest > 0:
                    time.sleep(rest)

        if viewer:
            try:
                import mujoco.viewer
                with mujoco.viewer.launch_passive(self.model, self.data) as v:
                    for _ in range(loops):
                        run_once(sync=v.sync)
                        if not v.is_running():
                            break
            except Exception as e:
                print(f"[player] viewer 不可用（{type(e).__name__}），改 headless 渲染。")
                for _ in range(loops):
                    run_once(sync=None)
        else:
            for _ in range(loops):
                run_once(sync=None)

        if renderer is not None and frames_png:
            self._save_gif(frames_png, os.path.join(record_dir, f"{motion.name}.gif"), motion.fps)
        return True

    def _headless(self, motion: RobotMotion, loops: int):
        """无 mujoco：干跑统计，证明帧流合法。"""
        b = motion.body
        print(f"[headless] '{motion.name}': {motion.n_frames} 帧 @ {motion.fps}fps "
              f"= {motion.duration:.2f}s, body shape={b.shape}")
        print(f"[headless] 关节角范围 min={b.min():+.3f} max={b.max():+.3f}, "
              f"n_joints={self.robot.n_joints}")
        if motion.positions is not None:
            print(f"[headless] FK 坐标: positions shape={motion.positions.shape}")
        return False

    @staticmethod
    def _save_gif(frames, path, fps):
        try:
            import imageio
            imageio.mimsave(path, frames, fps=fps)
            print(f"[player] 已保存 GIF：{path}")
        except ImportError:
            np.save(path.replace(".gif", ".npy"), np.array(frames))
            print(f"[player] 未装 imageio，帧已存为 .npy：{path.replace('.gif', '.npy')}")
