# PRD｜情绪驱动的机器人肢体动作（Emotion-Driven Motion）

| 项 | 内容 |
|----|------|
| **版本** | v0.2（生成式实时管线） |
| **作者** | 产品（PM） |
| **日期** | 2026-06-15 |
| **状态** | 待评审 |
| **关联** | `ARCHITECTURE.md`（机器人本体）、`.mujoco_cache_fullmesh_v5/`（仿真模型） |
| **变更** | v0.1 手工关键帧库 → **v0.2 改为「LLM 意图放大 → MLD 动作生成 → 重定向 → MuJoCo 注入」实时管线**（手工关键帧降级为兜底，见 §10） |

---

## 1. 背景与目标

### 1.1 现状
AI 对话已打通：模型在产出回复文本时会输出一个**情绪标签**。但情绪只停留在文字/语音层，
机器人本体（24 连杆 / 23 关节人形）是静止的，用户感受不到"它有情绪"。

### 1.2 目标
用一条**生成式动作管线**，把情绪/意图实时变成机器人的肢体动作：说"非常委屈"时，机器人能
低头、抬臂贴脸做出"揉眼睛"的姿态——动作不再是写死的几个，而是**由文本描述实时生成**。

### 1.3 一句话定义
> **情绪/意图文本 → LLM 放大成动作描述 → MLD 生成人体动作 → 重定向到 23 关节 → MuJoCo 实时播放。**

### 1.4 非目标（本期不做）
- ❌ 不做情绪识别（上游已给标签，本期只消费）
- ❌ 不做真机控制（先 MuJoCo 仿真）
- ❌ 不做行走/移动/平衡控制（漂浮基座下用 **qpos 运动学回放**，不跑物理，见 §4.3）
- ❌ 不自己训练动作生成模型（直接用开源 **MLD / MotionGPT** 预训练权重）

---

## 2. 用户故事

| 编号 | 角色 | 故事 |
|------|------|------|
| US-1 | 用户 | 我说了委屈的事，机器人**低头、抬手贴脸做"揉眼睛"**，像真的在委屈 |
| US-2 | 用户 | 我跟它打招呼，它**挥手**回应 |
| US-3 | 开发 | 我输入一句自然语言描述，机器人就能在仿真里**当场演出来**，不用手写关键帧 |
| US-4 | 开发 | 模型算出的动作即便超出关节限位，安全过滤器会**实时钳制**，仿真不会爆炸 |
| US-5 | 开发 | 人体骨架到机器人 23 关节的**重定向**有一套可复用脚本，新动作零额外成本 |

---

## 3. 情绪 → 意图描述（管线入口）

不再维护"写死的 6 个动作"，而是维护**6 种情绪 → 意图描述模板**。LLM 把情绪词"放大"成
MLD 能听懂的、可执行的英文动作描述（第一步）。

| 情绪标签 | 中文意图 | LLM 放大后的英文描述（喂给 MLD）示例 |
|----------|----------|--------------------------------------|
| `happy` | 开心 | "A person raises both arms outward and nods their head with the upper body leaning slightly forward." |
| `sad` | 难过 | "A person lowers their head, draws both shoulders inward, and shrinks the upper body slightly." |
| `angry` | 生气 | "A person leans the torso forward, presses both arms downward and tense, with the head slightly lowered." |
| `surprise` | 惊讶 | "A person quickly raises the head, opens both arms wide, and leans the upper body backward." |
| `fear` | 害怕 | "A person turns the head aside and pulls back, raising both arms to guard in front of the chest." |
| `neutral` | 待机 | "A person stands still and breathes calmly with tiny idle motions of the head." |
| `(意图)grieved` | 委屈 | "A person lowers their head slightly, brings both hands up to the face, and moves the wrists left and right near the eyes."（揉眼睛） |

> **意图放大（Intention Amplification）**：直接把"非常委屈"喂给 MLD，模型会懵。先用一个**微型 LLM**
> （Qwen-7B 级别本地模型，或直接 API）把情绪词翻译成"主体 + 部位 + 方向"结构的英文动作描述，
> 大幅提升 MLD 生成质量。`intensity` 可作为副词注入（"slightly / strongly"）。

---

## 4. 系统架构：四步实时管线

```
①情绪/意图        ②动作描述           ③人体动作流              ④机器人关节流
  "非常委屈"  ──►  微型LLM  ──►  "lower head, hands   ──►  MLD  ──►  24骨骼点 3~5s   ──►  重定向+限位  ──►  MuJoCo
              意图放大      to face, move wrists..."  (Latent Diffusion)  3D轨迹             Retargeting        qpos注入
```

| 步骤 | 模块 | 职责 | 输入 → 输出 |
|------|------|------|-------------|
| **① 意图放大** | 微型 LLM（Qwen-7B / API） | 情绪词 → 结构化英文动作描述 | `"非常委屈"` → 英文描述 |
| **② 动作生成** | **MLD（Motion Latent Diffusion）** | 英文描述 → 人体动作轨迹 | 文本 → 24 骨骼点 × 3~5s 序列 |
| **③ 重定向+限位** | Retarget 引擎（自研，**核心难点**） | 人体骨架 → 机器人 23 关节 + 安全钳制 | SMPL 轨迹 → `qpos` 帧流 |
| **④ 仿真注入** | MuJoCo 运行时服务 | 关节角流实时写入仿真 | 30fps `qpos` → 画面 |

### 4.1 为什么选 MLD
MLD 把扩散过程放在**潜空间（Latent Space）**，生成速度远快于像素/坐标空间扩散，几百毫秒即可
出一段 3~5s 动作，是清单里**最适合实时**的方案。输出为 HumanML3D/SMPL 体系的人体骨架轨迹。

### 4.2 第三步：重定向与限位保护【核心难点，最需要写代码】
MLD 生成的是"人类"动作，要翻译成机器人能执行的指令：

- **"揉眼睛"的硬件折算**：机器人**没有手指**，腕只有 1 个 `wrist_pitch`。重定向需把"手贴脸"
  折算到 `shoulder_pitch`（抬臂）+ `elbow_joint`（屈肘把前臂抬到脸侧）+ `wrist_pitch`（小幅往复），
  用**拳头/前臂末端**贴近脸部，再用腕/肘小幅左右模拟"揉"。
- **限位保护（安全过滤器）**：实时检查每一帧关节角，超出 `ARCHITECTURE.md §3.2` 限位立即钳制。
  例：肘关节限位 `[-1.571, 0.000]`，一旦 MLD 算出 -2.0，**强行卡死在 -1.571**，
  防止仿真里"骨折"/物理爆炸。`continuous` 关节（手腕、膝）做软限幅。

### 4.3 第四步：MuJoCo 注入方式【重要技术决策】
MuJoCo 运行时不重新加载模型，Python 脚本作为常驻服务，把第三步的关节角流（~30fps）实时写入。
**注入有两条路，本期明确选 A：**

| 方式 | 写入 | 前提 | 取舍 |
|------|------|------|------|
| **A. qpos 运动学回放**（✅ 本期选用） | `data.qpos` 直接设关节角 + `mj_forward` | 无需执行器 | 不跑接触动力学，**不会因漂浮基座/无平衡而摔倒**，最稳，先验证观感 |
| B. ctrl 物理驱动（下期） | `data.ctrl` 设目标位 | **需先给 MJCF 补 `<actuator>`** | 真实物理，但要调 PD 增益 + 平衡，复杂 |

> ⚠️ **必须知道的两个仓库事实**：
> 1. 当前 `fullmesh_v5.xml` **没有 `<actuator>` 段**，`data.ctrl` 现在是空的——直接用 ctrl 会失败。
>    走 B 路线**必须先给 MJCF 加位置执行器**。
> 2. 模型是**漂浮基座**（`root_free` freejoint）且腿无驱动，一跑物理就会倒。
>    本期用 qpos 回放正好绕开"没有平衡控制器"这个坑。

### 4.4 重定向方案的两代选型【技术路线】

重定向（人体骨架 → 机器人 23 关节）是本项目最大的不确定性。它有两代做法，**本期用第一代，
第二代作为升级路线**——这个分代思路来自 NVIDIA SONIC 论文（arXiv:2511.07820）的关键设计。

| | **第一代：手写映射（✅ 本期）** | **第二代：学习式隐式重定向（升级路线）** |
|---|---|---|
| 做法 | 人工写规则：人体关节 → 机器人关节的对应表 + 限位钳制 | 训一个编码器-解码器，把人体动作压进隐空间，解码器直接吐机器人关节角 |
| "揉眼睛"怎么处理 | 手工规定"手贴脸"→ 折算到 shoulder+elbow+wrist | 模型从数据里**自己学会**这种形态差异的折算 |
| 灵感来源 | 传统 retarget | **SONIC 的 Universal Token Space**：人体编码器 `ℰh` + 重建损失 `ℒrecon`，让编码-解码过程**本身就是"人→机器人"的隐式重定向器**，运行时无需显式映射 |
| 优点 | 简单、可控、可解释、零训练 | 动作自然、能处理大形态差异、一套通吃多模态 |
| 缺点 | 形态差异大时走样、每类动作都要调 | **要训练数据 + 算力**；SONIC 那套是 G1 专用、21000 GPU 小时级别 |

> **本期决策**：MVP **只做第一代手写映射**——6 种情绪的动作形态有限，手写映射 + 限位完全够用，
> 且零训练成本、出问题好调。
>
> **为什么记录第二代**：等动作种类暴增、或将来上标准机型（如 Unitree G1）时，
> SONIC 式的"学习式隐式重定向"是正解——它把"手写映射"这件最脏最累的活变成模型自动学。
> 那时若直接用 G1，SONIC（配 GEM 文本→动作生成器）几乎就是整条"情绪→文本→动作→执行"
> 管线的现成底座。**但它绑定 G1 本体、需重训，套不到当前自定义机器人上**（见 §9）。

### 4.5 对齐 GR00T-WBC / MotionBricks 的架构优化【借鉴落地】

读过 NVIDIA 官方代码库 `GR00T-WholeBodyControl`（含 SONIC、MotionBricks）后，下面 5 个**具体工程模式**
直接拿来优化我们的架构。每条都标了出处，便于对照。

**关键验证 ✅**：MotionBricks 的交互 demo（`motionbricks/scripts/interactive_demo_g1.py:74-97`）
驱动 MuJoCo 的核心就是 `mj_data.qpos[:] = qpos; mujoco.mj_forward(...)`——**纯 qpos 运动学回放，
不跑力控**。这与我们 §4.3 选的 A 路线完全一致，**官方自己的实时 demo 就是这么干的**，
说明 MVP 走 qpos 是被验证过的正确路线，不是妥协。

| # | 借鉴模式 | 出处 | 我们怎么落地 |
|---|----------|------|--------------|
| 1 | **统一骨架对象 + 显式 qpos 索引表** | `motionbricks/docs/motion_representation.md`（G1Skeleton34 + MuJoCo Joint Mapping） | 定义 `SouyanSkeleton23`：把 23 关节的树 + qpos 索引固化成一个对象，**一举消灭"三套命名"乱象**（见下方索引表） |
| 2 | **root / body 分离的运动表示** | 同上（root 5 维 + body 409 维分开） | 运动流拆成「根(浮动基座 7 维) + 23 身体关节」。情绪动作期**根保持固定/微动**，只让身体表达 |
| 3 | **6D 旋转表示关节** | 论文 + 代码 `geometry/`（6D continuous rotation） | Motion Player 内部用 6D 旋转存关节，插值平滑无翻转，**注入前再转成 MuJoCo hinge 角** |
| 4 | **逐帧 feature-vector 的统一运动格式** | `motion_representation.md`（每帧归一化特征向量） | Motion Player 消费的是「帧序列」而非临时关键帧；JSON 关键帧降级为**上层创作语法糖**，编译成帧流 |
| 5 | **验证过的实时控制循环** | `interactive_demo_g1.py`（get_next_frame → qpos 注入 → mj_forward → sync → sleep，自回归续帧） | 直接照搬这个循环骨架，含 `use_qpos` 开关对应我们的 A/B 路线 |

> **生成层的演进对齐**：MotionBricks 的生成是**模块化的**（VQVAE tokenizer + pose 模型 + root 模型，
> 即"smart primitives"）。我们 MVP 用单体 MLD，但**把生成层接口设计成"tokenizer/pose/root 三件套"
> 的形状**，将来换成 MotionBricks/SONIC 就是替换实现、而非重写管线。

#### SouyanSkeleton23 —— 本机器人 MuJoCo qpos 布局（立即可用）

`fullmesh_v5.xml` = 1 个浮动基座 + 23 个 hinge 关节，**qpos 共 30 维**。注入时按此索引写：

```
qpos[0:3]   根平移 (x, y, z)
qpos[3:7]   根四元数 (w, x, y, z)
qpos[7]     waist_yaw_joint          qpos[19]    neck_yaw_joint
qpos[8]     right_shoulder_pitch     qpos[20]    face_joint
qpos[9]     right_shoulder_roll      qpos[21]    waist_roll_joint
qpos[10]    right_shoulder_yaw       qpos[22]    right_hip_pitch_joint
qpos[11]    right_elbow_joint        qpos[23]    right_hip_roll_joint
qpos[12]    right_wrist_pitch        qpos[24]    right_knee_joint
qpos[13]    left_shoulder_pitch      qpos[25]    right_ankle_pitch_joint
qpos[14]    left_shoulder_roll       qpos[26]    left_hip_pitch_joint
qpos[15]    left_shoulder_yaw        qpos[27]    left_hip_roll_joint
qpos[16]    left_elbow_joint         qpos[28]    left_knee_joint
qpos[17]    left_wrist_pitch         qpos[29]    left_ankle_pitch_joint
qpos[18]    neck_pitch_joint
```
> 落地建议：**运行时从 `mj_model` 按关节名查索引**（别硬编码数字），名字→索引的映射就封装在
> `SouyanSkeleton23` 里——这正是 MotionBricks 用 skeleton 对象的做法，换模型版本也不会错位。

---

## 5. 数据与接口

### 5.1 意图描述规格（① 的输出 / ② 的输入）
结构化英文，建议含：主体(`A person`) + 身体部位 + 动作 + 方向/幅度副词。便于 MLD 稳定生成。

### 5.2 生成动作缓存（②→③ 之间，强烈建议）
同一情绪反复触发不必每次重算。把 MLD 生成结果**按意图文本哈希缓存**成机器人关节序列
（复用 v0.1 的 JSON 关键帧格式即可），常用情绪走缓存，新意图才实时生成——兼顾延迟与多样性。

### 5.3 运行期接口
```python
player.play_intent(text: str, intensity: float = 1.0)   # 主入口：一句话→生成→播放
player.play_emotion(emotion: str, intensity: float = 1.0) # 情绪标签→查§3模板→play_intent
player.interrupt()                                       # 打断，平滑回 neutral
player.is_playing() -> bool
```

---

## 6. 交互细节与边界

| 场景 | 处理 |
|------|------|
| 生成耗时 > 阈值 | 先播一个占位"待机微动"，生成完再切入；常用情绪走缓存（§5.2）规避 |
| 播放中来新情绪 | 平滑打断：从当前姿态插值到新动作首帧，不瞬移 |
| 重定向后自碰撞/穿模 | 限位钳制 + 碰撞规避（动作设计避免肢体交叠） |
| 标签不在 6 种内 | 回退 `neutral` + 记录未知标签 |
| MLD 出"诡异/抖动"动作 | 时间平滑滤波 + 限位钳制；缓存里淘汰坏样本 |
| MLD/LLM 服务异常 | 回退到 §10 的手工关键帧兜底库，不阻塞对话 |

---

## 7. 成功指标

| 指标 | 目标 | 说明 |
|------|------|------|
| **意图可辨识率**（盲测看动作猜意图） | ≥ 70% | 含"揉眼睛"等具体动作 |
| **端到端延迟**（缓存命中） | ≤ 300ms | 常用情绪走缓存 |
| **端到端延迟**（实时生成，未命中） | ≤ 1.5s | LLM 翻译 + MLD 生成 + 重定向，**比 v0.1 放宽** |
| **动作异常率**（穿模/超限/抖动） | ≤ 3% | 限位+平滑后统计 |
| **重定向覆盖**（人体动作能映射的比例） | 上半身 ≥ 90% | 手指动作必然丢失 |

---

## 8. 里程碑：分两阶段"闭眼通关"

> 不要一上来全连起来。先离线把"人体骨架→机器人关节"调通，再打通实时流。

### 阶段 1 — 离线调通映射（重点：retarget）
1. **先建 `SouyanSkeleton23`**（§4.5）：封装 23 关节名→qpos 索引、限位表、6D↔hinge 转换。一切的地基
2. 拿一段标准 SMPL/BVH 动捕（甚至不用 MLD），写重定向脚本：人体骨架 → souyan-robot 23 关节 + 限位钳制
3. 照搬 MotionBricks 的循环（`interactive_demo_g1.py`）：`qpos[:]=...` → `mj_forward` → `sync` 回放
4. 再接 **MLD / MotionGPT**：输入 `"wave hands"` → 生成 → 走同一条 retarget+回放链路
- **验收**：机器人在仿真里能把生成的"挥手"演完，无穿模/超限

### 阶段 2 — 打通实时流
1. 把 MLD 包成**本地 API 服务**
2. MuJoCo 运行脚本里加 `input()` 输入框：输入文字 → 异步请求 API → 数据塞进控制循环
3. 接 §3 的"情绪→意图描述"模板，对接上游情绪标签
- **验收**：端到端"说话→标签→生成→动作"跑通；常用情绪走缓存达标延迟

| 阶段 | 交付 | 关键风险 |
|------|------|----------|
| **阶段 1** | retarget 脚本 + qpos 回放 demo | 人体↔机器人骨架映射（核心难点） |
| **阶段 2** | MLD API + 实时注入 + 上游对接 | 实时延迟、生成质量稳定性 |

---

## 9. 风险与依赖

| 风险 | 影响 | 应对 |
|------|------|------|
| **重定向是核心难点**：人体 24 骨骼 ≠ 机器人 23 关节，无手指 | 动作走形/手部丢失 | 阶段 1 专门攻克；手部统一折算到肘+腕 |
| **MJCF 无执行器**，ctrl 不可用 | B 路线直接失败 | 本期走 qpos；要物理时先补 `<actuator>` |
| **漂浮基座无平衡**，跑物理即倒 | 无法用力控 | qpos 运动学回放绕开 |
| **实时延迟**：LLM+MLD+retarget 串联 | 体感卡顿 | 常用情绪缓存（§5.2）；异步占位动作 |
| **MLD 生成不稳定**（抖动/诡异） | 观感差 | 平滑滤波 + 限位 + 坏样本淘汰 |
| **collision=高面数网格**，自碰撞开销大 | 仿真卡 | 动作避免肢体交叠；后续换凸包碰撞体 |
| **三套命名**（语义名/`l*.STL`/`j*`）| 对接出错 | 引擎统一用语义关节名 + 映射表 |

### 兜底方案（Fallback）
v0.1 的**手工关键帧库**保留为兜底：MLD/LLM 服务不可用、或某情绪生成质量长期不达标时，
直接播手工动作，保证"任何时候机器人都有反应"。

---

## 10. 后续规划（Next）

1. **B 路线物理驱动**：给 MJCF 补位置执行器 → `data.ctrl` 力控，动作更真实
2. **真机执行后端**：同一管线驱动 j1..j25 舵机/电机
3. **平衡控制器**：上真机/上物理后，用 RL 让机器人站稳、能配合下肢动作
4. **手部升级**：若硬件加灵巧手，恢复"揉眼睛"等精细动作
5. **协同语音手势**：叠加 audio-to-gesture，让"说话时摆动"与"情绪 pose"融合
6. **学习式重定向（SONIC 路线）**：动作种类暴增后，用编码器-解码器替代手写映射（见 §4.4）
7. **整机底座升级**：若改用标准机型（Unitree G1 等），可直接对接 **SONIC + GEM** 作为
   "文本→动作生成 + 全身控制 + sim-to-real" 的现成底座，跳过自研重定向与平衡

---

*附：关节树与限位见 `ARCHITECTURE.md §3`；仿真模型用
`.mujoco_cache_fullmesh_v5/0403_souyan-robot_asm-3-23-21.fullmesh_v5.xml`。*
