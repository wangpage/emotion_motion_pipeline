// actions.js —— 情绪 → 任意机器人的连续自然动作（纯前端版）
// 与 Python 管线共用 ../config/emotions.json（单一真源）。动作用语义角色描述，
// 经 semantic.js 的映射落到当前加载机器人的真实关节。范围采样 + 二次运动 → 连续自然、每次略不同。

import { expandRole } from './semantic.js';

// ── 加载共享情绪库 ──────────────────────────────────────────────
let LIB = null;          // { fps, emotions:{...}, emotion_to_intent:{...} }
export async function loadLibrary(url = '../config/emotions.json') {
  const res = await fetch(url);
  LIB = await res.json();
  return LIB;
}
export function getLibrary() { return LIB; }
export function intentKeys() { return LIB ? Object.keys(LIB.emotions) : []; }

// ── 意图解析（中文口语关键词 → 意图键 + 强度，移植 intent.py 规则版）──
const KEYWORDS = [
  [['委屈', '揉眼', '抹眼', '哭'], 'grieved'],
  [['开心', '高兴', '快乐', '太好了', '棒', '开森'], 'happy'],
  [['难过', '伤心', '沮丧', '失落', '郁闷'], 'sad'],
  [['生气', '愤怒', '气', '火大', '恼'], 'angry'],
  [['惊讶', '吃惊', '震惊', '哇', '天呐', '居然'], 'surprise'],
  [['害怕', '恐惧', '怕', '吓', '担心'], 'fear'],
  [['撒娇'], 'coy'], [['害羞', '羞'], 'shy'],
  [['打招呼', '挥手', '你好', 'hi', 'hello', '再见', '拜拜', '嗨'], 'wave'],
  [['点头', '同意', '好的', '没错'], 'nod'],
  [['摇头', '不行', '不要', '不对'], 'shake'],
  [['待机', '中性', '平静', '站', 'idle', '休息'], 'neutral'],
];
const STRONG = ['非常', '特别', '超级', '极其', '好', '很', '太'];
const WEAK = ['有点', '稍微', '略', '一点', '些许'];

export function amplify(text) {
  const t = (text || '').toLowerCase();
  let key = 'neutral';
  for (const [kws, k] of KEYWORDS) { if (kws.some(w => t.includes(w.toLowerCase()))) { key = k; break; } }
  let intensity = 1.0;
  if (STRONG.some(w => text.includes(w))) intensity = 1.3;
  else if (WEAK.some(w => text.includes(w))) intensity = 0.7;
  const desc = (LIB && LIB.emotions[key]) ? key : 'neutral';
  return { raw: text, intent: key, emotion: key, intensity, desc };
}

// AIyizhizai 中文情绪词 → [意图键, 强度]（来自共享库）
export function emotionToIntent(emotionZh) {
  const m = (LIB && LIB.emotion_to_intent) || {};
  return m[emotionZh] || ['neutral', 1.0];
}

const EASE = {
  linear: t => t,
  ease_in: t => t * t,
  ease_out: t => 1 - (1 - t) ** 2,
  ease_in_out: t => 3 * t * t - 2 * t * t * t,
};

function sampleT(t) { return Array.isArray(t) ? t[0] + Math.random() * (t[1] - t[0]) : t; }
function sampleVal(spec) {
  if (spec && typeof spec === 'object') return (spec.c || 0) + (spec.s || 0) * (Math.random() * 2 - 1);
  return +spec || 0;
}

// 情绪 → 当前机器人的具体关键帧 [{t, joints:{jointName:val}, ez}]
function resolve(intentKey, mapping, intensity) {
  const emo = (LIB.emotions[intentKey]) || LIB.emotions.neutral || { keyframes: [] };
  const fps = emo.fps || LIB.fps || 30;
  const used = new Set();
  const out = [];
  let prevT = -1e9;
  for (const kf of emo.keyframes) {
    let t = sampleT(kf.t); t = Math.max(t, prevT + 1 / fps); prevT = t;
    const joints = {};
    for (const [role, spec] of Object.entries(kf.pose || {})) {
      const val = sampleVal(spec) * intensity;
      for (const rk of expandRole(role)) {
        const js = mapping[rk]; if (!js) continue;
        const [jname, sign] = js;
        joints[jname] = (joints[jname] || 0) + sign * val;
        used.add(rk);
      }
    }
    out.push({ t, joints, ez: kf.ez || 'ease_in_out' });
  }
  return { kfs: out, fps, used };
}

// ── 动画播放器：在当前 URDF 上实时回放（关键帧插值 + 二次运动）──
export class ActionPlayer {
  constructor(getRobot) {
    this.getRobot = getRobot;
    this.mapping = {};
    this._raf = null;
    this.playing = false;
    this.onUpdate = null;
    this.onState = null;
  }

  setMapping(mapping) { this.mapping = mapping || {}; }

  stop() { if (this._raf) cancelAnimationFrame(this._raf); this._raf = null; this.playing = false; }

  _clamp(robot, name, v) {
    const j = robot.joints[name]; if (!j) return v;
    if (j.jointType === 'continuous') return Math.max(-1.5708, Math.min(1.5708, v));
    const lo = j.limit?.lower, hi = j.limit?.upper;
    if (lo != null && hi != null && hi > lo) return Math.max(lo, Math.min(hi, v));
    return v;
  }

  play(intentKey, intensity = 1.0) {
    const robot = this.getRobot();
    if (!robot) { this.onState && this.onState('请先加载机器人模型'); return false; }
    if (!LIB) { this.onState && this.onState('情绪库未加载'); return false; }

    const { kfs, fps, used } = resolve(intentKey, this.mapping, intensity);
    if (!kfs.length || !used.size) {
      this.onState && this.onState(`⚠ 当前机器人没有可驱动 ${intentKey} 的关节`);
    }
    // 涉及关节并集 + 二次运动随机相位
    const names = new Set();
    kfs.forEach(k => Object.keys(k.joints).forEach(n => names.add(n)));
    const sec = {};
    names.forEach(n => sec[n] = {
      f1: 0.15 + Math.random() * 0.2, f2: 0.4 + Math.random() * 0.3,
      p1: Math.random() * 6.28, p2: Math.random() * 6.28,
    });
    // 展开完整向量（继承，从 0 开始）
    let prev = {}; names.forEach(n => prev[n] = 0);
    const full = kfs.map(k => { const m = { ...prev }; for (const n in k.joints) m[n] = k.joints[n]; prev = m; return { t: k.t, ez: k.ez, m }; });
    const dur = full[full.length - 1].t || 1;
    const AMP = 0.03, taper = 0.4;

    this.stop();
    this.playing = true;
    this.onState && this.onState(`▶ 执行：${intentKey}`);
    const start = performance.now();

    const tick = () => {
      const el = (performance.now() - start) / 1000;
      const tt = Math.min(el, dur);
      let s = 0; while (s < full.length - 1 && tt > full[s + 1].t) s++;
      const a = full[s], b = full[Math.min(s + 1, full.length - 1)];
      const span = b.t - a.t;
      const alpha = span <= 0 ? 1 : EASE[b.ez || 'ease_in_out']((tt - a.t) / span);
      // 二次运动淡入淡出包络
      let env = 1; if (tt < taper) env = tt / taper; else if (tt > dur - taper) env = Math.max(0, (dur - tt) / taper);
      names.forEach(n => {
        let v = (1 - alpha) * a.m[n] + alpha * b.m[n];
        const w = sec[n];
        v += AMP * env * (0.6 * Math.sin(6.283 * w.f1 * tt + w.p1) + 0.4 * Math.sin(6.283 * w.f2 * tt + w.p2));
        robot.setJointValue(n, this._clamp(robot, n, v));
      });
      this.onUpdate && this.onUpdate();
      if (el < dur) { this._raf = requestAnimationFrame(tick); }
      else { this.playing = false; this.onState && this.onState('✓ 完成'); }
    };
    this._raf = requestAnimationFrame(tick);
    return true;
  }
}
