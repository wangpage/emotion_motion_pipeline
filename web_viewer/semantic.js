// semantic.js —— JS 版语义映射（与 Python core/semantic.py 口径一致）
// 把已加载的 urdf-loader robot 的真实关节，自动映射到语义角色 + 方向符号。
// 输出 { mapping: {role_key:[jointName, sign]}, report:{source, mapped, unmapped} }

import * as THREE from 'three';

export const SIDES = ['R', 'L'];
export const SINGLE_ROLES = ['neck_pitch', 'neck_yaw', 'head_tilt', 'waist_yaw', 'waist_roll', 'waist_pitch'];
export const PAIRED_ROLES = ['shoulder_pitch', 'shoulder_abduct', 'shoulder_yaw', 'elbow_flex',
  'wrist_pitch', 'hip_pitch', 'hip_abduct', 'hip_yaw', 'knee_flex', 'ankle_pitch'];

const ROLE_DOF = {
  neck_pitch: 'pitch', neck_yaw: 'yaw', head_tilt: 'roll',
  waist_yaw: 'yaw', waist_roll: 'roll', waist_pitch: 'pitch',
  shoulder_pitch: 'pitch', shoulder_abduct: 'roll', shoulder_yaw: 'yaw',
  elbow_flex: 'pitch', wrist_pitch: 'pitch',
  hip_pitch: 'pitch', hip_abduct: 'roll', hip_yaw: 'yaw', knee_flex: 'pitch', ankle_pitch: 'pitch',
};
const PART_ROLES = {
  neck: ['neck_pitch', 'neck_yaw', 'head_tilt'], head: ['head_tilt', 'neck_pitch', 'neck_yaw'],
  waist: ['waist_yaw', 'waist_roll', 'waist_pitch'], torso: ['waist_yaw', 'waist_roll', 'waist_pitch'],
  spine: ['waist_pitch', 'waist_yaw', 'waist_roll'],
  shoulder: ['shoulder_pitch', 'shoulder_abduct', 'shoulder_yaw'], elbow: ['elbow_flex'],
  wrist: ['wrist_pitch'], hip: ['hip_pitch', 'hip_abduct', 'hip_yaw'], knee: ['knee_flex'], ankle: ['ankle_pitch'],
};

// 展开情绪库角色键：成对 -> R./L. 双侧；单体/已带前缀 -> 原样
export function expandRole(role) {
  if (SINGLE_ROLES.includes(role)) return [role];
  if (role.includes('.')) {
    const [side, base] = role.split('.');
    return (SIDES.includes(side) && PAIRED_ROLES.includes(base)) ? [role] : [];
  }
  if (PAIRED_ROLES.includes(role)) return SIDES.map(s => `${s}.${role}`);
  return [];
}

// ── souyan 内置 profile（与 Python SOUYAN_PROFILE 完全一致）──
const SOUYAN_PROFILE = {
  neck_pitch: ['neck_pitch_joint', 1], neck_yaw: ['neck_yaw_joint', 1], head_tilt: ['face_joint', 1],
  waist_yaw: ['waist_yaw_joint', 1], waist_roll: ['waist_roll_joint', 1],
  'R.shoulder_pitch': ['right_shoulder_pitch_joint', 1], 'L.shoulder_pitch': ['left_shoulder_pitch_joint', 1],
  'R.shoulder_abduct': ['right_shoulder_roll_joint', 1], 'L.shoulder_abduct': ['left_shoulder_roll_joint', -1],
  'R.shoulder_yaw': ['right_shoulder_yaw_joint', 1], 'L.shoulder_yaw': ['left_shoulder_yaw_joint', -1],
  'R.elbow_flex': ['right_elbow_joint', -1], 'L.elbow_flex': ['left_elbow_joint', -1],
  'R.wrist_pitch': ['right_wrist_pitch_joint', 1], 'L.wrist_pitch': ['left_wrist_pitch_joint', -1],
  'R.hip_pitch': ['right_hip_pitch_joint', 1], 'L.hip_pitch': ['left_hip_pitch_joint', 1],
  'R.hip_abduct': ['right_hip_roll_joint', 1], 'L.hip_abduct': ['left_hip_roll_joint', -1],
  'R.knee_flex': ['right_knee_joint', -1], 'L.knee_flex': ['left_knee_joint', -1],
  'R.ankle_pitch': ['right_ankle_pitch_joint', 1], 'L.ankle_pitch': ['left_ankle_pitch_joint', 1],
};

const PART_RE = /shoulder|elbow|wrist|neck|head|face|waist|torso|spine|hip|knee|ankle/i;
const DOF_RE = /pitch|roll|yaw/i;

function namePart(n) {
  const m = PART_RE.exec(n); if (!m) return null;
  const p = m[0].toLowerCase();
  return ({ face: 'head', torso: 'waist', spine: 'waist' })[p] || p;
}
function nameSide(n) {
  const s = n.toLowerCase();
  if (/(^|[^a-z])right/.test(s) || s.startsWith('r_') || /_r(_|$)/.test(s)) return 'R';
  if (/(^|[^a-z])left/.test(s) || s.startsWith('l_') || /_l(_|$)/.test(s)) return 'L';
  return null;
}
function nameDof(n) { const m = DOF_RE.exec(n); return m ? m[0].toLowerCase() : null; }

const AXIS_DOF = ['roll', 'pitch', 'yaw'];   // dominant x/y/z（URDF 原生 Z-up 框架）
const ANATO = { pitch: 1, roll: 1, yaw: 1 };

function roleFor(part, dof) {
  const roles = PART_ROLES[part] || [];
  for (const r of roles) if (ROLE_DOF[r] === dof) return r;
  return roles[0] || null;
}
function signFor(role, dof, side, ab, joint) {
  // 屈曲类按限位可用方向（ROM 更大一侧）定号，保证任何 URDF 都弯得起来
  if ((role === 'elbow_flex' || role === 'knee_flex') && joint
      && joint.limit && joint.limit.lower != null && joint.limit.upper != null) {
    return Math.abs(joint.limit.upper) >= Math.abs(joint.limit.lower) ? 1 : -1;
  }
  const di = { roll: 0, pitch: 1, yaw: 2 }[dof];
  const base = ab[di] >= 0 ? 1 : -1;
  const mirror = (side === 'L' && (dof === 'roll' || dof === 'yaw')) ? -1 : 1;
  const flex = (role === 'elbow_flex' || role === 'knee_flex') ? -1 : 1;
  return base * ANATO[dof] * mirror * flex;
}

// 关节轴在“机器人根坐标系”下的方向（不受显示用 up-axis 旋转影响）
function axisInRoot(robot, joint) {
  const a = new THREE.Vector3(joint.axis.x, joint.axis.y, joint.axis.z);
  const qJoint = joint.getWorldQuaternion(new THREE.Quaternion());
  const qRoot = robot.getWorldQuaternion(new THREE.Quaternion());
  // axis(root) = qRoot^-1 * qJoint * axis(local)
  a.applyQuaternion(qJoint).applyQuaternion(qRoot.invert());
  return [a.x, a.y, a.z];
}

function matchProfile(robot) {
  const names = new Set(Object.keys(robot.joints));
  const sig = ['right_shoulder_pitch_joint', 'neck_pitch_joint', 'waist_yaw_joint'];
  if (sig.every(s => names.has(s))) {
    const mapping = {};
    for (const [rk, v] of Object.entries(SOUYAN_PROFILE)) if (names.has(v[0])) mapping[rk] = v;
    return { mapping, source: 'profile:souyan' };
  }
  return null;
}

function heuristic(robot) {
  const mapping = {};
  // 复位到中性，保证读到的世界朝向对应 angle=0
  const movable = Object.values(robot.joints).filter(j => j.jointType !== 'fixed');
  movable.forEach(j => robot.setJointValue(j.name, 0));
  robot.updateMatrixWorld(true);
  for (const j of movable) {
    const part = namePart(j.name); if (!part) continue;
    const ab = axisInRoot(robot, j);
    let dof = nameDof(j.name);
    if (!dof) { let mi = 0; for (let i = 1; i < 3; i++) if (Math.abs(ab[i]) > Math.abs(ab[mi])) mi = i; dof = AXIS_DOF[mi]; }
    let side = nameSide(j.name);
    if (!side) { const p = j.getWorldPosition(new THREE.Vector3()); side = p.y < -1e-3 ? 'R' : (p.y > 1e-3 ? 'L' : null); }
    const role = roleFor(part, dof); if (!role) continue;
    const paired = PAIRED_ROLES.includes(role);
    const key = paired ? `${side}.${role}` : role;
    if (paired && !side) continue;
    if (mapping[key]) continue;
    mapping[key] = [j.name, signFor(role, dof, side, ab, j)];
  }
  return { mapping, source: 'heuristic' };
}

export function buildSemanticMap(robot) {
  const res = matchProfile(robot) || heuristic(robot);
  const used = new Set(Object.values(res.mapping).map(v => v[0]));
  const all = Object.values(robot.joints).filter(j => j.jointType !== 'fixed').map(j => j.name);
  return {
    mapping: res.mapping,
    report: { source: res.source, mapped: Object.keys(res.mapping).length, unmapped: all.filter(n => !used.has(n)) },
  };
}
