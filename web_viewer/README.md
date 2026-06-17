# web_viewer —— 机器人查看器 + 一句话生成动作

纯前端（Three.js + urdf-loader），复刻 viewer.robotsfan.com 的核心体验，并打通
emotion_motion_pipeline：**上传模型 + 说一句话 → 机器人当场执行动作。**
默认加载本仓库的 souyan-robot。

## 运行

必须用 HTTP 服务（`file://` 无法 fetch 网格）。一键脚本（自动切到项目根起服务）：

```bash
./serve.sh
# 浏览器打开： http://localhost:8000/emotion_motion_pipeline/web_viewer/
```

或手动在**项目根目录**起：`python3 -m http.server 8000`。
为什么在根目录起：默认模型路径 `../../urdf/0403_*.urdf` + URDF 里网格 `../meshes/*.STL`，
从根目录起服务这些相对路径才解析得到。

## 一句话 → 动作（核心）

底部输入条：
- **文字**：输入"非常委屈地揉眼睛 / 我好开心 / 你好挥挥手"→ 回车 / 执行 ▶。
- **🎤 语音**：点麦克风说话（中文，Web Speech API，建议 Chrome）→ 自动识别并执行。
- **预设按钮**：开心/难过/生气/惊讶/害怕/委屈/挥手/点头/摇头 一键触发。

原理（移植自 `../pipeline`）：一句话 → 关键词意图解析（`actions.js:amplify`）→
关键帧动作库 → 实时插值驱动 URDF 关节（限位钳制）。纯前端，无需后端。

## 功能

| 区域 | 功能 |
|------|------|
| 画布 | 左键旋转 / 右键平移 / 滚轮缩放，网格地面 + 阴影 + 灯光 |
| 工具栏 | 视觉·质心·坐标轴·关节轴·阴影·光照 开关；+Z/+Y 上轴；面板显隐；主题 |
| 文件 | 拖拽 / 按钮 加载 `.urdf/.xml` + `meshes`（支持选文件夹） |
| 关节 | 每个可动关节一个滑条，弧度/角度切换、重置、限位显示，实时驱动模型 |
| 结构 | links/joints 树，👁 单独显隐 |
| 编辑 | 显示模型源码，可下载 |

## 技术说明

- **用 URDF 而非 MJCF**：原站解析 MJCF，浏览器里从零写 MJCF 解析器很重。本仓库有等价的
  `urdf/0403_*.urdf`（同机器人、同网格、同关节），用成熟的 `urdf-loader` 更可靠。
  加载自定义 MJCF 暂不支持（可作为后续扩展）。
- 依赖走 CDN（jsdelivr）：three@0.160 + urdf-loader@0.12.6，见 `index.html` 的 importmap。
  首次打开需联网拉 CDN。
- 关节限位、轴向来自 URDF；`continuous` 关节滑条范围用 ±π。
