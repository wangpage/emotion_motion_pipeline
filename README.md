# emotion_motion_pipeline —— 上传任意机器人 URDF，一句话/情绪 → 连续自然的肢体动作

把 **AI 对话输出的情绪文字**，变成 **任意机器人（你上传的 URDF）的连续、自然的全关节动作**。
机器人本体由 URDF 描述（关节、轴、限位），管线自动理解它、把情绪映射到它的真实关节上，
输出**每关节的角度轨迹 + 正运动学 3D 坐标**；动作来自**范围采样 + 二次微动**，所以是一段
**连续、每次略不同的自然动作，而不是一个僵硬的静止姿态**。

```
 上传 URDF ──► 我和 AI 对话 ──► AI 吐情绪文字 ──► 本项目接管：
 (机器人本体)                    "开心/委屈/..."
        │                              │
        ▼                              ▼
  ⓪ 理解URDF+语义映射 → ① 意图放大 → ② 动作生成(范围采样+二次运动) → ③ (重定向) → ④ 回放/导出
  关节/轴/限位/树           情绪→意图     角色→真实关节, 连续自然轨迹       人体→机器人    MuJoCo / JSON
        └──────────────── 同一套引擎，Python 与 web_viewer 共用 ───────────────┘
```

## 核心设计：语义角色层（机器人无关）

不同机器人的关节命名千差万别。情绪库**不写具体关节名**，而是写**语义角色**
（`shoulder_pitch` / `shoulder_abduct`(外展) / `elbow_flex`(屈肘) / `neck_pitch` / `waist_roll` …）。
上传 URDF 后，`SemanticMapper` 用**关节名 + 轴在机体系下的方向 + 运动树位置**自动把真实关节
归类到这些角色，并给每个角色一个 `sign`，让“角色值为正”在任何机器人上都表示同一个解剖动作。

- **已知机器人（souyan）**：命中**内置 profile**（从已在 MuJoCo 标定过的动作库反推），100% 精确。
- **任意机器人**：走**自动启发式**，尽力识别；对不上的关节自动跳过、保持中性。
  > ⚠️ 通用路径的方向符号是 best-effort，**以仿真/画面验收为准**（换轴极易把方向搞反，PRD 早有提醒）。

“范围”怎么体现自然：①情绪库每个目标是 `{c:中心, s:幅度}`，运行时在 `[c-s, c+s]` 内采样
→ 峰值姿态每次不同；②整段叠加低频**二次运动（呼吸/微摆）**并首尾淡入淡出。两者共同让同一情绪
每次播放都是一段**连续、自然、不重复**的轨迹。

## 30 秒跑起来（零大模型、无需 GPU）

```bash
cd emotion_motion_pipeline
pip install -r requirements.txt            # 至少 numpy；装 mujoco 才有窗口

# 默认机器人（自带 assets/souyan.urdf）
python run.py "我好开心啊" --no-viewer
python run.py "非常委屈地揉眼睛" --save out/grieved.json   # JSON 含关节角 + FK 坐标

# 任意机器人：上传它的 URDF
python run.py "你好挥挥手" --urdf path/to/your_robot.urdf --no-viewer
python run.py --emotion happy --urdf path/to/your_robot.urdf
```

没装 `mujoco` 也能跑：进入 **headless 干跑**，打印帧流/FK 统计，证明 ⓪①②③ 正确。
装了 `mujoco` 且提供匹配的 MJCF（`--mjcf`）就能看到机器人动起来。

## 目录结构

```
emotion_motion_pipeline/
├── run.py                  # CLI 入口：--urdf 上传机器人；串起 ⓪①②③④
├── core/                   # ★ 机器人无关的统一引擎
│   ├── roles.py            #   语义角色词表 + 正方向约定
│   ├── robot.py            #   RobotModel：解析任意 URDF（关节/轴/限位/树）+ 限位钳制 + 最小 FK
│   └── semantic.py         #   SemanticMapper：真实关节 → 角色 + 符号（souyan profile / 自动启发式）
├── pipeline/
│   ├── intent.py           # ① 意图放大（规则 / Claude）
│   ├── library.py          #   读情绪库 + 范围采样 + 角色→真实关节解析
│   ├── generate.py         # ② 动作生成（procedural / mock / mld）+ 二次运动 + FK
│   ├── retarget.py         # ③ 重定向：SMPL-22 → 角色 → 任意机器人（次要路径）
│   ├── motion.py           #   RobotMotion：关节角轨迹 + FK 坐标 + JSON
│   └── player.py           # ④ MuJoCo 注入回放（按关节名解析 qpos）
├── config/emotions.json    # ★ 角色化 + 带范围的情绪库（Python 与 web_viewer 共用的单一真源）
├── web_viewer/             # 纯前端：上传 URDF + 说一句话/AI情绪 → 浏览器里当场执行
│   ├── semantic.js         #   JS 版语义映射（与 core/semantic.py 同口径）
│   ├── actions.js          #   fetch 共享情绪库 + 范围采样 + 二次运动 + 驱动 URDF
│   └── app.js              #   Three.js + urdf-loader；加载即建语义映射
├── assets/souyan.urdf      # 自带的 souyan 本体（默认机器人 / 测试用）
└── skeleton/               # 遗留：souyan 专用骨架，仅 scripts/render_poses.py 标定工具用
```

## 输出格式（`--save` 的 JSON）

```json
{
  "name": "happy", "fps": 30,
  "joint_names": ["waist_yaw_joint", ...],   // 该机器人的真实关节，顺序 = body 的列
  "body": [[...], ...],                       // (T, n_joints) 每关节角度轨迹（连续）
  "link_names": ["pelvis", ...],
  "positions": [[[x,y,z], ...], ...],         // (T, n_links, 3) FK 算出的每连杆 3D 坐标
  "meta": {"roles_used": [...], "mapping_source": "profile:souyan", "clamped": 0}
}
```

## web_viewer（浏览器实时演示）

```bash
cd web_viewer && ./serve.sh          # 从项目根起 HTTP 服务
# 打开 http://localhost:8000/emotion_motion_pipeline/web_viewer/
```

- **上传任意 URDF**：拖拽 `.urdf` + meshes，或“加载文件/文件夹”。加载后顶部状态条会显示语义识别报告
  （识别到几个角色、几个关节未归类）。
- **一句话 → 动作**：输入“我好开心 / 非常委屈地揉眼睛 / 你好挥挥手”，或点预设按钮 / 🎤 语音。
- **接 AIyizhizai**：AI 对话吐情绪标签（开心/撒娇/害羞/担心/生气/平静…），经 BroadcastChannel /
  postMessage / `window.playEmotion(情绪)` 进来，自动在**当前加载的机器人**上播放连续动作。

> 默认机器人优先用同级 `urdf/`，取不到时回退项目内 `assets/souyan.urdf`（meshes 缺失不影响关节/动作逻辑）。

## 接入真 LLM 意图放大 / 真·一句话生成（MLD）

```bash
export ANTHROPIC_API_KEY=sk-...
python run.py "今天被夸了好开心" --intent-backend llm     # 无 key 自动回退规则版
```

MLD（`pipeline/generate.py:MLDGenerator`）留好注入点：提供 `mld_fn(text)->(T,22,3)` 的 SMPL-22 序列，
即走 `retarget.py`（SMPL→角色→任意机器人）。未接入时自动回退 procedural，保证管线不断。

## 已知边界

- 通用机器人的方向符号是 best-effort（屈肘/屈膝已按限位可用方向自动定号）；看着反了，
  在 `core/semantic.py` 加该机器人的 profile，或调 `config/emotions.json` 的角色值。
- 物理力控（`data.ctrl` + 平衡控制器）不在本期，本期是 A 路线 = 运动学 qpos 回放。
- `assets/souyan.urdf` 的 meshes 在同级 `meshes/`；Python 管线不依赖 mesh（只用关节/轴/限位/树）。
