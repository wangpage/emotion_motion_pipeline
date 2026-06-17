#!/usr/bin/env python3
"""
姿态渲染器 —— 给定关节角，渲染机器人静态姿态 PNG，用于"对着渲染图调方向"。

用法（需 mujoco+imageio，用 /usr/bin/python3）：
    /usr/bin/python3 scripts/render_poses.py find-front      # 多方位中性图，找正面机位
    /usr/bin/python3 scripts/render_poses.py emotions        # 渲染 6 情绪峰值姿态
    /usr/bin/python3 scripts/render_poses.py calib           # 单关节正向激励标定

输出到 out/render/。
"""
from __future__ import annotations
import os, sys
import numpy as np
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skeleton.souyan_skeleton import SouyanSkeleton23, JOINT_NAMES  # noqa
from pipeline.generate import keyframes_for  # noqa

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
MJCF = os.path.join(ROOT, ".mujoco_cache_fullmesh_v5",
                    "0403_souyan-robot_asm-3-23-21.fullmesh_v5.xml")
OUT = os.path.join(HERE, "out", "render")
os.makedirs(OUT, exist_ok=True)

sk = SouyanSkeleton23()
model = mujoco.MjModel.from_xml_path(MJCF)
data = mujoco.MjData(model)
QIDX = sk.resolve_qpos_indices(model)

# 上半身中心附近作为 lookat（torso 在 x≈-0.08, z≈0.3）
LOOKAT = np.array([-0.08, 0.0, 0.30])


def set_pose(joints: dict):
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0  # 四元数 w=1
    for name, val in joints.items():
        data.qpos[QIDX[JOINT_NAMES.index(name)]] = sk.clamp_angle(name, float(val))
    mujoco.mj_forward(model, data)


def render(azimuth=180.0, elevation=-10.0, distance=1.6, w=420, h=480):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = LOOKAT
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    r = mujoco.Renderer(model, height=h, width=w)
    r.update_scene(data, camera=cam)
    img = r.render().copy()
    r.close()
    return img


def grid(images, cols, path, labels=None):
    h, w, _ = images[0].shape
    rows = (len(images) + cols - 1) // cols
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, im in enumerate(images):
        rr, cc = divmod(i, cols)
        canvas[rr*h:(rr+1)*h, cc*w:(cc+1)*w] = im
    imageio.imwrite(path, canvas)
    print("wrote", path, "labels:", labels)


def peak_joints(intent: str) -> dict:
    """取该意图关键帧里非空关节最多的那一帧作为'峰值姿态'。"""
    kfs, _ = keyframes_for(intent, 1.0)
    best = max(kfs, key=lambda kf: len(kf.get("joints", {})))
    return best["joints"]


def cmd_find_front():
    set_pose({})
    imgs = [render(azimuth=a) for a in (0, 90, 180, 270)]
    grid(imgs, 4, os.path.join(OUT, "find_front.png"),
         labels=["az0", "az90", "az180", "az270"])


def cmd_emotions():
    names = ["happy", "sad", "angry", "surprise", "fear", "neutral"]
    imgs = []
    for n in names:
        set_pose(peak_joints(n))
        imgs.append(render())
    grid(imgs, 3, os.path.join(OUT, "emotions_peak.png"), labels=names)
    # 单张也存，便于细看
    for n in names:
        set_pose(peak_joints(n))
        imageio.imwrite(os.path.join(OUT, f"emo_{n}.png"), render())


def cmd_calib():
    """单关节正向激励：看 +0.6rad 时各关节往哪动。"""
    tests = [
        ("R_sh_pitch+", {"right_shoulder_pitch_joint": 0.9}),
        ("L_sh_pitch+", {"left_shoulder_pitch_joint": 0.9}),
        ("R_sh_roll+",  {"right_shoulder_roll_joint": 0.9}),
        ("L_sh_roll+",  {"left_shoulder_roll_joint": 0.9}),
        ("R_elbow-",    {"right_elbow_joint": -1.2}),
        ("L_elbow-",    {"left_elbow_joint": -1.2}),
        ("neck_pitch+", {"neck_pitch_joint": 0.5}),
        ("neck_yaw+",   {"neck_yaw_joint": 0.6}),
        ("waist_roll+", {"waist_roll_joint": 0.5}),
    ]
    imgs = []
    for _, j in tests:
        set_pose(j)
        imgs.append(render())
    grid(imgs, 3, os.path.join(OUT, "calib.png"), labels=[t[0] for t in tests])


def cmd_sagittal():
    """侧视近景：俯仰类关节 +/- 各渲一张，清楚看抬头/低头、抬臂前后。"""
    AZ = 270  # 侧视
    tests = [
        ("neutral",        {}),
        ("neck_pitch +0.5", {"neck_pitch_joint": 0.5}),
        ("neck_pitch -0.5", {"neck_pitch_joint": -0.5}),
        ("R_sh_pitch +1.2", {"right_shoulder_pitch_joint": 1.2}),
        ("R_sh_pitch -0.7", {"right_shoulder_pitch_joint": -0.7}),
        ("waist_roll +0.5", {"waist_roll_joint": 0.5}),
    ]
    imgs = []
    for _, j in tests:
        set_pose(j)
        imgs.append(render(azimuth=AZ, elevation=-5, distance=1.4))
    grid(imgs, 3, os.path.join(OUT, "sagittal.png"), labels=[t[0] for t in tests])


def cmd_roll():
    """前视：左右 shoulder_roll 的 +/- 对照，确认'外展/内收'符号。"""
    tests = [
        ("R_roll +0.8", {"right_shoulder_roll_joint": 0.8}),
        ("R_roll -0.8", {"right_shoulder_roll_joint": -0.8}),
        ("wide R+ L-",  {"right_shoulder_roll_joint": 0.8, "left_shoulder_roll_joint": -0.8}),
        ("L_roll +0.8", {"left_shoulder_roll_joint": 0.8}),
        ("L_roll -0.8", {"left_shoulder_roll_joint": -0.8}),
        ("in R- L+",    {"right_shoulder_roll_joint": -0.8, "left_shoulder_roll_joint": 0.8}),
    ]
    imgs = []
    for _, j in tests:
        set_pose(j)
        imgs.append(render())
    grid(imgs, 3, os.path.join(OUT, "roll.png"), labels=[t[0] for t in tests])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "find-front"
    {"find-front": cmd_find_front, "emotions": cmd_emotions,
     "calib": cmd_calib, "sagittal": cmd_sagittal, "roll": cmd_roll}[cmd]()
