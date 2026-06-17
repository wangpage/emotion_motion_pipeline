import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { ColladaLoader } from 'three/addons/loaders/ColladaLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import URDFLoader from 'urdf-loader';
import { amplify, ActionPlayer, loadLibrary, emotionToIntent } from './actions.js';
import { buildSemanticMap } from './semantic.js';

// 全局错误横幅：任何 JS 错误/未处理 Promise 都显示在页面顶部，便于排查
function showErrorBanner(msg) {
  let el = document.getElementById('err-banner');
  if (!el) {
    el = document.createElement('div'); el.id = 'err-banner';
    el.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:9999;background:#b00020;color:#fff;'
      + 'font:13px/1.5 monospace;padding:8px 12px;white-space:pre-wrap;max-height:40vh;overflow:auto';
    document.body.appendChild(el);
  }
  el.textContent = '⚠ 页面脚本出错（功能可能失效）：\n' + msg;
}
window.addEventListener('error', e => showErrorBanner(e.message + (e.filename ? `\n@ ${e.filename}:${e.lineno}` : '')));
window.addEventListener('unhandledrejection', e => showErrorBanner('Promise: ' + (e.reason?.message || e.reason)));

// 默认模型：仓库根的 souyan-robot URDF（web_viewer 在 emotion_motion_pipeline/ 下，故上溯两级）
const DEFAULT_URDF = '../../urdf/0403_souyan-robot_asm-3-23-21.urdf';
const FALLBACK_URDF = '../assets/souyan.urdf';   // 项目内自包含副本（同级 urdf/ 缺失时回退）

// ---------- 场景 ----------
const viewport = document.getElementById('viewport');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x3a3d42);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(1.6, 1.3, 2.0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0.7, 0);

// 灯光
const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6);
scene.add(hemi);
const key = new THREE.DirectionalLight(0xffffff, 1.1);
key.position.set(3, 5, 4); key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.near = 0.5; key.shadow.camera.far = 20;
key.shadow.camera.left = -3; key.shadow.camera.right = 3;
key.shadow.camera.top = 3; key.shadow.camera.bottom = -3;
scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.3);
fill.position.set(-3, 2, -2); scene.add(fill);

// 地面网格 + 阴影承接面
const grid = new THREE.GridHelper(10, 20, 0x888888, 0x555555);
grid.material.opacity = 0.35; grid.material.transparent = true;
scene.add(grid);
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(10, 10),
  new THREE.ShadowMaterial({ opacity: 0.25 })
);
ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true;
scene.add(ground);

const worldAxes = new THREE.AxesHelper(0.5); worldAxes.visible = false;
scene.add(worldAxes);

// ---------- 状态 ----------
let robot = null;
let upAxis = 'z';
let useDegrees = false;
let showLimits = true;
const jointAxisHelpers = [];
const comMarkers = [];

// ---------- URDF 加载 ----------
const manager = new THREE.LoadingManager();
// 网格是异步加载的：URDF 结构解析完时几何体还没到，首帧 frameRobot() 会按空包围盒算出错误相机。
// 等本批网格全部加载完成后再对准一次，模型就稳稳落在画面里。
manager.onLoad = () => { if (robot) frameRobot(); };
const loader = new URDFLoader(manager);
const MESH_MAT = () => new THREE.MeshStandardMaterial({ color: 0xc9ccd1, metalness: 0.25, roughness: 0.65 });
let meshStats = { ok: 0, fail: 0 };   // 每次加载的网格统计

// 按扩展名选 loader，支持 STL / OBJ / DAE / GLTF/GLB；urlResolver 把 mesh 路径映射到本地 blob
function makeMeshCb(urlResolver) {
  return (path, mgr, done) => {
    const url = urlResolver ? urlResolver(path) : path;
    const ext = (url.split('?')[0].split('.').pop() || '').toLowerCase();
    const ok = (obj) => { meshStats.ok++; obj.traverse?.(o => { if (o.isMesh) { o.castShadow = o.receiveShadow = true; } }); if (obj.isMesh) { obj.castShadow = obj.receiveShadow = true; } done(obj); };
    const fail = (err) => { meshStats.fail++; console.warn('网格加载失败', path, '->', url, err); done(null, err); };
    try {
      if (ext === 'stl') new STLLoader(mgr).load(url, g => { g.computeVertexNormals(); ok(new THREE.Mesh(g, MESH_MAT())); }, undefined, fail);
      else if (ext === 'obj') new OBJLoader(mgr).load(url, o => { o.traverse(c => { if (c.isMesh) c.material = MESH_MAT(); }); ok(o); }, undefined, fail);
      else if (ext === 'dae') new ColladaLoader(mgr).load(url, c => ok(c.scene), undefined, fail);
      else if (ext === 'glb' || ext === 'gltf') new GLTFLoader(mgr).load(url, g => ok(g.scene), undefined, fail);
      else fail(new Error('不支持的网格格式: .' + ext));
    } catch (e) { fail(e); }
  };
}
loader.loadMeshCb = makeMeshCb(null);   // 默认：按原始路径加载（用于 URL 加载默认模型）

function clearRobot() {
  if (robot) { scene.remove(robot); robot = null; }
  jointAxisHelpers.length = 0; comMarkers.length = 0;
}

function applyUpAxis() {
  if (!robot) return;
  // URDF 通常 Z-up；three 是 Y-up。Z-up 时绕 X 转 -90°让其站立。
  robot.rotation.set(upAxis === 'z' ? -Math.PI / 2 : 0, 0, 0);
}

function loadURDFFromURL(url, label) {
  loader.load(url, result => {
    clearRobot();
    robot = result;
    robot.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    scene.add(robot);
    applyUpAxis();
    frameRobot();
    buildJointPanel();
    buildStructureTree();
    buildComMarkers();
    buildMap();
    document.getElementById('center-actions').classList.remove('hidden');
    document.getElementById('files-body').innerHTML =
      `<div class="loaded">📦 ${label || url.split('/').pop()}</div>`;
    fetchSource(url, label);
  }, undefined, err => {
    console.error('URDF 加载失败', err);
    // 默认 souyan 在同级 urdf/ 取不到时，回退项目内自包含副本（meshes 缺失不影响关节/动作逻辑）
    if (url !== FALLBACK_URDF) { loadURDFFromURL(FALLBACK_URDF, 'souyan.urdf (assets)'); return; }
    document.getElementById('files-body').insertAdjacentHTML('beforeend',
      `<div class="muted" style="color:#ff8a8a">加载失败：${err?.message || err}</div>`);
  });
}

function frameRobot() {
  const box = new THREE.Box3().setFromObject(robot);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  controls.target.copy(center);
  const maxDim = Math.max(size.x, size.y, size.z);
  camera.position.set(center.x + maxDim * 0.9, center.y + maxDim * 0.4, center.z + maxDim * 1.6);
  controls.update();
}

// ---------- 关节面板 ----------
function buildJointPanel() {
  const body = document.getElementById('joints-body');
  body.innerHTML = '';
  const joints = Object.values(robot.joints).filter(j => j.jointType !== 'fixed');
  if (!joints.length) { body.innerHTML = '<div class="muted">无可动关节</div>'; return; }

  joints.forEach(j => {
    const lower = j.limit.lower ?? -Math.PI;
    const upper = j.limit.upper ?? Math.PI;
    const cont = j.jointType === 'continuous';
    const lo = cont ? -Math.PI : lower, hi = cont ? Math.PI : upper;
    const el = document.createElement('div');
    el.className = 'joint';
    el.innerHTML = `
      <div class="jname">${j.name}</div>
      <div class="jrow">
        <span class="lim lo"></span>
        <input type="range" min="${lo}" max="${hi}" step="0.001" value="${j.angle || 0}">
        <span class="lim hi"></span>
        <input class="val" type="text">
        <span class="unit"></span>
      </div>`;
    const range = el.querySelector('input[type=range]');
    const val = el.querySelector('.val');
    const loEl = el.querySelector('.lo'), hiEl = el.querySelector('.hi'), unit = el.querySelector('.unit');
    const fmt = v => useDegrees ? (v * 180 / Math.PI).toFixed(0) : (+v).toFixed(2);
    const refresh = () => {
      const a = j.angle || 0;
      range.value = a; val.value = fmt(a);
      loEl.textContent = showLimits ? fmt(lo) : '';
      hiEl.textContent = showLimits ? fmt(hi) : '';
      unit.textContent = useDegrees ? '°' : 'rad';
    };
    range.addEventListener('input', () => { robot.setJointValue(j.name, +range.value); val.value = fmt(+range.value); });
    val.addEventListener('change', () => {
      let v = parseFloat(val.value); if (isNaN(v)) return;
      if (useDegrees) v = v * Math.PI / 180;
      robot.setJointValue(j.name, v); range.value = v;
    });
    el._refresh = refresh; refresh();
    body.appendChild(el);
  });
}
function refreshJoints() { document.querySelectorAll('#joints-body .joint').forEach(e => e._refresh && e._refresh()); }

// ---------- 语义映射：理解上传的 URDF，把情绪角色落到它的真实关节 ----------
function buildMap() {
  if (!robot) return;
  const { mapping, report } = buildSemanticMap(robot);
  player.setMapping(mapping);
  const statusEl = document.getElementById('action-status');
  const tag = report.source.startsWith('profile') ? '内置标定' : '自动识别';
  const msg = `🧠 ${tag}：识别 ${report.mapped} 个动作角色`
    + (report.unmapped.length ? `，${report.unmapped.length} 个关节未归类` : '，全部关节已归类');
  if (statusEl) { statusEl.textContent = msg; statusEl.classList.remove('ok'); }
  console.log('[semantic]', report.source, mapping);
}

// ---------- 结构树 ----------
function buildStructureTree() {
  const body = document.getElementById('structure-body');
  body.innerHTML = '';
  const root = document.createElement('div'); root.className = 'tree';
  const walk = (obj, container) => {
    obj.children.forEach(c => {
      const isLink = c.isURDFLink, isJoint = c.isURDFJoint;
      if (!isLink && !isJoint) { walk(c, container); return; }
      const node = document.createElement('div'); node.className = 'node';
      const row = document.createElement('div'); row.className = 'row';
      row.innerHTML = `<span class="tg">${c.children.length ? '▸' : '·'}</span>
        <span class="${isJoint ? 'jt' : 'ln'}">${isJoint ? '⚙ ' + c.name : '🔗 ' + c.name}</span>
        <span class="eye">👁</span>`;
      row.querySelector('.eye').addEventListener('click', e => {
        e.stopPropagation(); c.visible = !c.visible;
        row.querySelector('.eye').style.opacity = c.visible ? .4 : .15;
      });
      node.appendChild(row); container.appendChild(node);
      walk(c, node);
    });
  };
  walk(robot, root);
  body.appendChild(root);
}

// ---------- 质心标记 ----------
function buildComMarkers() {
  comMarkers.length = 0;
  robot.traverse(o => {
    if (o.isURDFLink) {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 12),
        new THREE.MeshBasicMaterial({ color: 0xff5a5a }));
      m.visible = false; o.add(m); comMarkers.push(m);
    }
  });
}

// ---------- 关节轴 ----------
function buildJointAxes(show) {
  jointAxisHelpers.forEach(h => h.parent && h.parent.remove(h));
  jointAxisHelpers.length = 0;
  if (!show || !robot) return;
  Object.values(robot.joints).forEach(j => {
    if (j.jointType === 'fixed') return;
    const dir = new THREE.Vector3(j.axis.x, j.axis.y, j.axis.z).normalize();
    const arrow = new THREE.ArrowHelper(dir, new THREE.Vector3(0, 0, 0), 0.12, 0x2f7bff, 0.04, 0.025);
    j.add(arrow); jointAxisHelpers.push(arrow);
  });
}

// ---------- XML 源码 ----------
async function fetchSource(url, label) {
  try {
    const txt = await (await fetch(url)).text();
    showSource(txt, label || url.split('/').pop());
  } catch (e) { showSource('// 无法读取源码: ' + e, label); }
}
function showSource(txt, name) {
  document.getElementById('src-name').textContent = name || 'model';
  const st = document.getElementById('src-status'); st.textContent = '已保存'; st.className = 'badge saved';
  const code = document.getElementById('editor-code');
  code.textContent = txt.split('\n').slice(0, 400).map((l, i) => String(i + 1).padStart(3, ' ') + '  ' + l).join('\n');
  window._srcText = txt;
}

// ---------- 工具栏 ----------
const toggles = {
  visual: v => robot && robot.traverse(o => { if (o.isMesh && !o._isMarker) o.visible = v; }),
  collision: () => {},   // URDF 视觉=碰撞同网格，占位
  inertia: () => {},
  com: v => comMarkers.forEach(m => m.visible = v),
  axes: v => { worldAxes.visible = v; },
  jointaxes: v => buildJointAxes(v),
  shadow: v => { renderer.shadowMap.enabled = v; key.castShadow = v;
    robot && robot.traverse(o => { if (o.isMesh) o.castShadow = v; }); ground.visible = v; },
  light: v => { key.intensity = v ? 1.1 : 0.0; fill.intensity = v ? 0.3 : 0.0; hemi.intensity = v ? 0.6 : 0.25; },
};
document.querySelectorAll('[data-toggle]').forEach(btn => {
  btn.addEventListener('click', () => {
    btn.classList.toggle('active');
    const on = btn.classList.contains('active');
    const fn = toggles[btn.dataset.toggle]; fn && fn(on);
  });
});

// 面板显隐
document.querySelectorAll('[data-panel]').forEach(btn => {
  btn.addEventListener('click', () => {
    btn.classList.toggle('active');
    document.getElementById('panel-' + btn.dataset.panel)
      .classList.toggle('hidden', !btn.classList.contains('active'));
  });
});

// 关节面板控件
document.getElementById('rad-btn').addEventListener('click', e => {
  useDegrees = false; e.target.classList.add('active'); document.getElementById('deg-btn').classList.remove('active'); refreshJoints();
});
document.getElementById('deg-btn').addEventListener('click', e => {
  useDegrees = true; e.target.classList.add('active'); document.getElementById('rad-btn').classList.remove('active'); refreshJoints();
});
document.getElementById('limit-btn').addEventListener('click', e => {
  showLimits = !showLimits; e.target.classList.toggle('active', showLimits); refreshJoints();
});
document.getElementById('reset-joints').addEventListener('click', resetPose);
document.getElementById('reset-pose')?.addEventListener('click', resetPose);
function resetPose() {
  if (!robot) return;
  Object.values(robot.joints).forEach(j => j.jointType !== 'fixed' && robot.setJointValue(j.name, 0));
  refreshJoints();
}

document.getElementById('upaxis').addEventListener('change', e => { upAxis = e.target.value; applyUpAxis(); });
document.getElementById('theme-btn').addEventListener('click', () => document.body.classList.toggle('light'));
document.getElementById('help-btn').addEventListener('click', () => alert(
  '左键拖动: 旋转\n右键拖动: 平移\n滚轮: 缩放\n关节面板: 拖动滑条控制关节\n拖拽 .urdf/.xml + meshes 文件夹加载自定义模型'));
document.getElementById('lang-btn').addEventListener('click', () => alert('Language switch is a stub in this clone.'));
document.getElementById('download-src').addEventListener('click', () => {
  if (!window._srcText) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([window._srcText], { type: 'text/xml' }));
  a.download = document.getElementById('src-name').textContent || 'model.urdf'; a.click();
});
document.getElementById('reload-src').addEventListener('click', () => loadURDFFromURL(DEFAULT_URDF));

// 面板最大化/关闭 + 拖动
document.querySelectorAll('.panel').forEach(p => {
  p.querySelector('[data-max]')?.addEventListener('click', () => p.classList.toggle('maxed'));
  p.querySelector('[data-close]')?.addEventListener('click', () => {
    p.classList.add('hidden');
    const id = p.id.replace('panel-', '');
    document.querySelector(`[data-panel="${id}"]`)?.classList.remove('active');
  });
  makeDraggable(p, p.querySelector('header'));
});
function makeDraggable(panel, handle) {
  if (!handle) return;
  let sx, sy, ox, oy, drag = false;
  handle.addEventListener('mousedown', e => {
    if (e.target.closest('.actions,.seg')) return;
    drag = true; sx = e.clientX; sy = e.clientY;
    const r = panel.getBoundingClientRect(); ox = r.left; oy = r.top;
    panel.style.right = 'auto'; panel.style.bottom = 'auto';
    e.preventDefault();
  });
  window.addEventListener('mousemove', e => {
    if (!drag) return;
    panel.style.left = (ox + e.clientX - sx) + 'px';
    panel.style.top = (oy + e.clientY - sy) + 'px';
  });
  window.addEventListener('mouseup', () => drag = false);
}

// ---------- 文件加载 ----------
document.getElementById('load-file').addEventListener('click', () => { console.log('[load-file] 打开文件选择框'); document.getElementById('file-input').click(); });
document.getElementById('load-folder').addEventListener('click', () => document.getElementById('folder-input').click());
document.getElementById('file-input').addEventListener('change', e => loadFromFiles([...e.target.files]));
document.getElementById('folder-input').addEventListener('change', e => loadFromFiles([...e.target.files]));

function loadFromFiles(files) {
  const fb = document.getElementById('files-body');
  if (!files.length) return;
  // 挑模型文件：优先真正的 .urdf，跳过 .mujoco_cache/* 与 .bak；其次才退到 .xml
  const rel = f => (f.webkitRelativePath || f.name || '');
  const clean = f => !/(mujoco_cache|\.bak$|node_modules)/i.test(rel(f));
  const urdfs = files.filter(f => /\.urdf$/i.test(f.name) && clean(f));
  const xmls = files.filter(f => /\.xml$/i.test(f.name) && clean(f));
  const modelFile = urdfs[0] || xmls[0] || files.find(f => /\.(urdf|xml)$/i.test(f.name));
  if (!modelFile) {
    fb.innerHTML = `<div class="muted" style="color:#ff8a8a">没找到 .urdf/.xml 模型文件。请用「加载文件夹」选包含 urdf 和 meshes 的整个目录。已选 ${files.length} 个文件。</div>`;
    return;
  }
  console.log('[loadFromFiles] 模型文件:', rel(modelFile), '| 候选 urdf:', urdfs.map(rel), '| 总文件', files.length);
  // 文件名(基名) → blob URL 映射，重写 mesh 路径；文件夹上传时也能匹配子目录里的网格
  const blobs = {};
  files.forEach(f => { blobs[(f.name || '').toLowerCase()] = URL.createObjectURL(f); });
  const resolver = (path) => {
    const base = (path.split('/').pop() || '').toLowerCase();
    return blobs[base] || path;
  };
  const meshCount = files.filter(f => /\.(stl|obj|dae|glb|gltf)$/i.test(f.name)).length;
  const reader = new FileReader();
  reader.onerror = () => { fb.innerHTML = `<div class="muted" style="color:#ff8a8a">读取 ${modelFile.name} 失败</div>`; };
  reader.onload = () => {
    try {
      const text = reader.result;
      meshStats = { ok: 0, fail: 0 };
      loader.loadMeshCb = makeMeshCb(resolver);
      clearRobot();
      robot = loader.parse(text);
      scene.add(robot); applyUpAxis(); frameRobot();
      buildJointPanel(); buildStructureTree(); buildComMarkers(); buildMap();
      document.getElementById('center-actions').classList.remove('hidden');
      const nJoints = Object.values(robot.joints).filter(j => j.jointType !== 'fixed').length;
      // 网格异步加载，稍后汇报成功/失败数
      setTimeout(() => {
        const meshMsg = meshCount === 0
          ? '未随附网格文件（关节/动作仍可用，模型可能不可见）'
          : `网格 ${meshStats.ok} 成功 / ${meshStats.fail} 失败（共 ${meshCount} 个）`;
        fb.innerHTML = `<div class="loaded">📦 ${modelFile.name}<br/><span class="muted">${nJoints} 个可动关节 · ${meshMsg}</span></div>`;
      }, 600);
      loader.loadMeshCb = makeMeshCb(null);   // 复位默认
      showSource(text, modelFile.name);
    } catch (e) {
      fb.innerHTML = `<div class="muted" style="color:#ff8a8a">解析 URDF 失败：${e.message}</div>`;
      console.error('URDF parse error', e);
    }
  };
  reader.readAsText(modelFile);
}

// 拖放：支持拖入整个文件夹（递归子目录拿到 urdf + meshes）
const overlay = document.getElementById('drop-overlay');
window.addEventListener('dragover', e => { e.preventDefault(); overlay.classList.remove('hidden'); });
window.addEventListener('dragleave', e => { if (e.clientX === 0 && e.clientY === 0) overlay.classList.add('hidden'); });
window.addEventListener('drop', async e => {
  e.preventDefault(); overlay.classList.add('hidden');
  // 优先用 FileSystem Entry API 递归文件夹；浏览器拖文件夹时 dataTransfer.files 不含子目录内容
  const items = e.dataTransfer.items;
  const canRecurse = items && items.length && typeof items[0].webkitGetAsEntry === 'function';
  if (canRecurse) {
    const entries = [...items].map(it => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
    const files = [];
    await Promise.all(entries.map(en => walkEntry(en, files)));
    if (files.length) { loadFromFiles(files); return; }
  }
  const files = [...e.dataTransfer.files]; if (files.length) loadFromFiles(files);
});

// 递归遍历拖入的文件/目录树，把每个 File 收进 out（并记录相对路径，便于诊断）
function walkEntry(entry, out) {
  return new Promise(resolve => {
    if (entry.isFile) {
      entry.file(f => {
        try { if (!('webkitRelativePath' in f) || !f.webkitRelativePath) {
          Object.defineProperty(f, 'webkitRelativePath', { value: entry.fullPath.replace(/^\//, ''), configurable: true });
        } } catch (_) {}
        out.push(f); resolve();
      }, () => resolve());
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      const all = [];
      const readBatch = () => reader.readEntries(batch => {
        if (!batch.length) { Promise.all(all.map(en => walkEntry(en, out))).then(resolve); return; }
        all.push(...batch); readBatch();   // readEntries 一次最多返回 ~100 项，需循环读完
      }, () => resolve());
      readBatch();
    } else { resolve(); }
  });
}

// ---------- 渲染循环 ----------
function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h);
}
window.addEventListener('resize', resize); resize();
(function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); })();

// ---------- 一句话 → 动作 ----------
const player = new ActionPlayer(() => robot);
player.onUpdate = refreshJoints;                       // 动画每帧刷新滑条
const statusEl = document.getElementById('action-status');
player.onState = msg => { statusEl.textContent = msg; statusEl.classList.toggle('ok', msg.startsWith('✓')); };

const actionInput = document.getElementById('action-input');

function runSentence(text) {
  if (!text || !text.trim()) return;
  const spec = amplify(text);
  statusEl.textContent = `识别意图：${spec.intent}（强度 ${spec.intensity}）`;
  statusEl.classList.remove('ok');
  player.play(spec.intent, spec.intensity);
}
document.getElementById('action-run').addEventListener('click', () => runSentence(actionInput.value));
actionInput.addEventListener('keydown', e => { if (e.key === 'Enter') runSentence(actionInput.value); });
document.querySelectorAll('#action-chips .chip').forEach(c =>
  c.addEventListener('click', () => { actionInput.value = c.textContent.trim(); player.play(c.dataset.intent, 1.0); }));
document.getElementById('sim-btn')?.addEventListener('click', () => runSentence(actionInput.value || '待机'));

// 语音输入（Web Speech API，中文）
const micBtn = document.getElementById('mic-btn');
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SR) {
  const rec = new SR();
  rec.lang = 'zh-CN'; rec.interimResults = false; rec.maxAlternatives = 1;
  let recording = false;
  micBtn.addEventListener('click', () => {
    if (recording) { rec.stop(); return; }
    try { rec.start(); recording = true; micBtn.classList.add('rec'); statusEl.textContent = '🎙 听你说…'; }
    catch (e) { statusEl.textContent = '语音启动失败：' + e.message; }
  });
  rec.onresult = e => {
    const text = e.results[0][0].transcript;
    actionInput.value = text; runSentence(text);
  };
  rec.onerror = e => { statusEl.textContent = '语音识别出错：' + e.error; };
  rec.onend = () => { recording = false; micBtn.classList.remove('rec'); };
} else {
  micBtn.addEventListener('click', () => statusEl.textContent = '当前浏览器不支持语音识别，请用文字输入（建议 Chrome）');
}

// ---------- 与 AIyizhizai 对接：情绪标签 → 自动动作 ----------
// 接收三种来源：BroadcastChannel（同源标签页）/ postMessage（iframe或window.open）/ 全局函数。
function onEmotion(emotionZh) {
  if (!emotionZh) return;
  const [intent, intensity] = emotionToIntent(emotionZh);
  actionInput.value = `〔AI情绪〕${emotionZh}`;
  statusEl.textContent = `🤖 收到 AIyizhizai 情绪：${emotionZh} → ${intent}`;
  statusEl.classList.remove('ok');
  player.play(intent, intensity);
}
window.playEmotion = onEmotion;                      // 供任何嵌入方直接调用
window.addEventListener('message', e => {            // iframe / window.open 跨源
  const d = e.data;
  if (d && d.type === 'emotion') onEmotion(d.emotion);
});
try {                                                // 同源多标签页
  const bus = new BroadcastChannel('aiyizhizai-emotion');
  bus.onmessage = e => onEmotion(e.data && e.data.emotion);
  statusEl.textContent = '✓ 已就绪，等待 AIyizhizai 情绪…';
} catch (e) { /* 老浏览器无 BroadcastChannel，靠 postMessage */ }

// 启动：先加载共享情绪库，再加载默认机器人（库就绪后预设按钮/对话才能驱动）
// 支持 ?urdf=<served-path> 直接指定机器人（带网格），免改代码 / 免上传。
const _qUrdf = new URLSearchParams(location.search).get('urdf');
loadLibrary()
  .then(() => { statusEl.textContent = '✓ 情绪库已加载，等待 AIyizhizai 情绪…'; })
  .catch(e => { statusEl.textContent = '情绪库加载失败：' + e.message + '（请用 HTTP 服务打开）'; })
  .finally(() => loadURDFFromURL(_qUrdf || DEFAULT_URDF,
    _qUrdf ? _qUrdf.split('/').pop() : '0403_souyan-robot_asm-3-23-21.urdf'));
