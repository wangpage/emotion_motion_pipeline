#!/usr/bin/env python3
"""
一句话/情绪 → 任意机器人的连续自然动作 —— 端到端 pipeline 入口。

    python run.py "非常委屈地揉眼睛"                 # 默认 souyan
    python run.py "我好开心啊" --no-viewer
    python run.py "你好呀" --urdf path/to/robot.urdf  # 任意机器人 URDF
    python run.py --emotion happy --save out/happy.json
    python run.py "挥挥手" --record out/

四步：意图放大 → (理解 URDF + 语义映射) → 动作生成(范围采样+二次运动) → (重定向) → 回放/导出。
输出含每关节角度轨迹 + 正运动学 3D 坐标（--save 的 JSON 里）。
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.intent import make_amplifier, ActionSpec, TEMPLATES, EMOTION_OF  # noqa: E402
from pipeline.generate import make_generator  # noqa: E402
from pipeline.library import EmotionLibrary  # noqa: E402
from pipeline.player import MujocoPlayer, DEFAULT_MJCF  # noqa: E402
from core.robot import RobotModel  # noqa: E402
from core.semantic import SemanticMapper  # noqa: E402

DEFAULT_URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "souyan.urdf")


def main():
    ap = argparse.ArgumentParser(description="一句话/情绪 → 任意机器人的连续自然动作")
    ap.add_argument("text", nargs="?", default="", help="一句话，如 '非常委屈地揉眼睛'")
    ap.add_argument("--emotion", help="直接给情绪/意图键（happy/sad/grieved/wave...）跳过文本解析")
    ap.add_argument("--urdf", default=DEFAULT_URDF, help="机器人本体 URDF（缺省 souyan）")
    ap.add_argument("--mjcf", default=None, help="MuJoCo 回放用的 MJCF（缺省 souyan fullmesh_v5）")
    ap.add_argument("--intent-backend", default="auto", choices=["auto", "rule", "llm"])
    ap.add_argument("--backend", default="procedural", choices=["procedural", "mock", "mld"])
    ap.add_argument("--no-viewer", action="store_true", help="不开 MuJoCo 窗口")
    ap.add_argument("--loops", type=int, default=2, help="循环播放次数")
    ap.add_argument("--seed", type=int, default=None, help="固定随机种子（默认每次不同=自然变化）")
    ap.add_argument("--save", help="把生成的 RobotMotion（含 FK 坐标）存成 JSON")
    ap.add_argument("--record", help="渲染并保存 GIF 到该目录（需 mujoco+imageio）")
    args = ap.parse_args()

    # --- Stage 0: 理解 URDF + 语义映射 ---
    robot = RobotModel.from_file(args.urdf)
    mapping, report = SemanticMapper().map(robot)
    print("=" * 64)
    print(f"⓪ 机器人    : {robot}")
    print(f"   {report.summary()}")

    lib = EmotionLibrary.load()

    # --- Stage 1: 意图放大 ---
    if args.emotion:
        key = args.emotion
        spec = ActionSpec(raw_text=f"--emotion {key}", emotion=EMOTION_OF.get(key, "intent"),
                          intent_key=key, description_en=TEMPLATES.get(key, ""), intensity=1.0)
    else:
        if not args.text:
            print("用法: python run.py \"一句话\"  或  --emotion happy")
            return 1
        spec = make_amplifier(args.intent_backend).amplify(args.text)

    if not lib.has(spec.intent_key):
        print(f"   ⚠ 情绪库无 '{spec.intent_key}'，回退 neutral")
    print(f"① 输入      : {spec.raw_text}")
    print(f"② 意图/情绪 : {spec.intent_key}  (emotion={spec.emotion}, intensity={spec.intensity})")
    print(f"③ 英文描述  : {spec.description_en}")

    # --- Stage 2/3: 生成（范围采样 + 二次运动 + FK） ---
    gen = make_generator(args.backend, robot, mapping=mapping, library=lib, seed=args.seed)
    motion = gen.generate(spec)
    print(f"④ 生成动作  : backend={motion.meta.get('backend')}, "
          f"{motion.n_frames} 帧 / {motion.duration:.2f}s, 限位钳制 {motion.meta.get('clamped', 0)} 次")
    if motion.meta.get("roles_used"):
        print(f"   涉及角色  : {', '.join(motion.meta['roles_used'])}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        motion.to_json(args.save)
        print(f"   已保存    : {args.save}（含 joint angles + FK positions）")

    # --- Stage 4: 播放 ---
    print("=" * 64)
    player = MujocoPlayer(robot, mjcf_path=args.mjcf)
    player.play(motion, viewer=not args.no_viewer, loops=args.loops, record_dir=args.record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
