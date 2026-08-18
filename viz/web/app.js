// Offline replay viewer for sentient-chicken. Loads a recorded trajectory (flat
// float32 binary + JSON metadata, written by `run.record`) and plays it back as a
// stylised 3D scene. The sim itself is 2D (`World.pos` is `(H, 2)`) -- world x/y map
// to three.js x/z, "up" is added purely for legibility, not simulated.
import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';

const CALL_FLOOR = 0.0759;      // sigmoid(REST_BIAS); a silent hen still reads this
const CALL_ON = CALL_FLOOR + 0.15; // threshold above the floor to count as "calling"
const CALL_COLORS = [0xe0e0e0, 0x5ad86b, 0xff4d4d, 0xffb020]; // contact/food/aerial/ground
const STRUCK_FLASH_S = 0.4;

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------
const container = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x12141a);
scene.fog = new THREE.Fog(0x12141a, 25, 60);

const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
camera.position.set(14, 16, 20);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(10, 0, 10);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x30281c, 0.9));
const sun = new THREE.DirectionalLight(0xfff4e0, 1.4);
sun.position.set(12, 22, 8);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
scene.add(sun);

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

// Floor + coop bounds are rebuilt per run (size can vary), everything else below is
// created once at the max scale ever seen and just made invisible when unused.
let floor = null, fence = null;
function buildFloor(size) {
  if (floor) scene.remove(floor, fence);
  const g = new THREE.PlaneGeometry(size, size, 20, 20);
  g.rotateX(-Math.PI / 2);
  floor = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: 0x3a3226, roughness: 1 }));
  floor.position.set(size / 2, 0, size / 2);
  floor.receiveShadow = true;
  scene.add(floor);

  const pts = [[0, 0], [size, 0], [size, size], [0, size], [0, 0]]
    .map(([x, z]) => new THREE.Vector3(x, 0.4, z));
  fence = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: 0x556 }));
  scene.add(fence);
  controls.target.set(size / 2, 0, size / 2);
  camera.position.set(size * 0.7, size * 0.8, size);
}

// ---------------------------------------------------------------------------
// Hens: one InstancedMesh, GPU-instanced so this stays cheap as flock size grows.
// ---------------------------------------------------------------------------
const HEN_GEOM = new THREE.ConeGeometry(0.22, 0.5, 8);
HEN_GEOM.rotateX(-Math.PI / 2);  // apex now points along local -Z, which is the axis
                                 // Object3D.lookAt() aims -- so the cone's point faces
                                 // whatever `dummy.lookAt()` is given below.
// No translate here, deliberately: the geometry stays centred on the object's own
// origin, which is also the pivot `dummy.rotation.x` (the head-down tilt below)
// rotates around. Baking a +Y offset into the geometry used to move the visual
// shape away from that pivot -- rotating a shape around a point well below its own
// centre swings the far end down, and at up to 0.5 rad tilt (active ~64% of the
// time, per the project's own head-down measurements) that pushed hens visibly
// through the floor. The lift now happens at HEN_PIVOT_Y below, applied to the
// pivot itself, so the tilt rotates the model around its own centre instead.
let henMesh = null, henCallMesh = null;
const dummy = new THREE.Object3D();
let flashUntil = null;   // per-hen wall-clock deadline for a struck flash
let lastStruck = null;   // per-hen cumulative strike-event count, previous frame

function buildHens(n) {
  if (henMesh) scene.remove(henMesh, henCallMesh);
  henMesh = new THREE.InstancedMesh(
    HEN_GEOM, new THREE.MeshStandardMaterial({ vertexColors: false }), n);
  henMesh.castShadow = true;
  henMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
  scene.add(henMesh);

  henCallMesh = new THREE.InstancedMesh(
    new THREE.SphereGeometry(0.09, 8, 8),
    new THREE.MeshBasicMaterial({ toneMapped: false }), n);
  henCallMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
  scene.add(henCallMesh);

  flashUntil = new Float32Array(n).fill(-1);
  lastStruck = new Float32Array(n).fill(NaN);
}

const HEN_COLOR = new THREE.Color(0xd9c9a3);
const HEN_HEAD_DOWN = new THREE.Color(0x9c8a63);
const HEN_STRUCK = new THREE.Color(0xff2a2a);
const HIDDEN = new THREE.Matrix4().makeScale(0, 0, 0);
// Height of the hen model's rotation pivot above the floor. Must clear the model's
// own half-extent (radius 0.22) plus how far its length (0.25, half of the 0.5
// cone height) swings down at the maximum head-down tilt (0.5 rad): 0.22*cos(0.5) +
// 0.25*sin(0.5) ~= 0.31, so 0.35 leaves a small margin rather than sitting exactly
// on the boundary.
const HEN_PIVOT_Y = 0.35;

// ---------------------------------------------------------------------------
// Food / water: static positions per run, only `food_amount` animates.
// ---------------------------------------------------------------------------
let foodMesh = null, waterMesh = null;
function buildResources(foodPos, waterPos) {
  if (foodMesh) scene.remove(foodMesh, waterMesh);
  foodMesh = new THREE.InstancedMesh(
    new THREE.CylinderGeometry(0.35, 0.4, 0.15, 10),
    new THREE.MeshStandardMaterial({ color: 0x7a9a3d }), foodPos.length);
  foodMesh.receiveShadow = true;
  for (let i = 0; i < foodPos.length; i++) {
    dummy.position.set(foodPos[i][0], 0.075, foodPos[i][1]);
    dummy.rotation.set(0, 0, 0); dummy.scale.set(1, 1, 1); dummy.updateMatrix();
    foodMesh.setMatrixAt(i, dummy.matrix);
  }
  scene.add(foodMesh);

  waterMesh = new THREE.InstancedMesh(
    new THREE.CylinderGeometry(0.4, 0.4, 0.06, 16),
    new THREE.MeshStandardMaterial({ color: 0x3a7bd5, metalness: 0.2, roughness: 0.1 }),
    waterPos.length);
  for (let i = 0; i < waterPos.length; i++) {
    dummy.position.set(waterPos[i][0], 0.03, waterPos[i][1]);
    dummy.rotation.set(0, 0, 0); dummy.scale.set(1, 1, 1); dummy.updateMatrix();
    waterMesh.setMatrixAt(i, dummy.matrix);
  }
  scene.add(waterMesh);
}

// ---------------------------------------------------------------------------
// Predators
// ---------------------------------------------------------------------------
const hawkMesh = new THREE.Mesh(
  new THREE.ConeGeometry(0.5, 1.4, 4), new THREE.MeshStandardMaterial({ color: 0x8a2a2a }));
hawkMesh.rotateX(Math.PI);
hawkMesh.castShadow = true;
scene.add(hawkMesh);

const foxMesh = new THREE.Mesh(
  new THREE.ConeGeometry(0.4, 0.9, 6), new THREE.MeshStandardMaterial({ color: 0xb85c1e }));
foxMesh.rotateX(Math.PI / 2);
foxMesh.castShadow = true;
scene.add(foxMesh);

// ---------------------------------------------------------------------------
// Run loading
// ---------------------------------------------------------------------------
let meta = null, arrays = null;   // arrays: name -> Float32Array view (per-frame stride baked in)
let frameIdx = 0, playing = true;

const runSelect = document.getElementById('runSelect');
const descEl = document.getElementById('desc');
const scrub = document.getElementById('scrub');
const clockEl = document.getElementById('clock');
const playBtn = document.getElementById('playBtn');
const speedSel = document.getElementById('speed');
const emptyEl = document.getElementById('empty');

async function fetchRuns() {
  const res = await fetch('/api/runs');
  return res.json();
}

function fmtClock(s) {
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}

async function loadRun(id) {
  const [m, bin] = await Promise.all([
    fetch(`/api/runs/${id}/meta.json`).then(r => r.json()),
    fetch(`/api/runs/${id}/trajectory.bin`).then(r => r.arrayBuffer()),
  ]);
  meta = m;
  arrays = {};
  for (const [name, info] of Object.entries(m.layout)) {
    const n = info.shape.reduce((a, b) => a * b, 1);
    arrays[name] = new Float32Array(bin, info.offset, n);
  }
  descEl.textContent = `${m.hens} hens · ${m.minutes} min · ${m.plastic ? 'plastic' : 'innate'}`
    + (m.description ? ` · ${m.description}` : '');

  buildFloor(m.coop_size);
  buildHens(m.hens);
  buildResources(m.food_pos, m.water_pos);

  scrub.max = String(m.n_frames - 1);
  frameIdx = 0;
  scrub.value = '0';
  render(0);
}

function stride(name) { return meta.layout[name].shape.slice(1).reduce((a, b) => a * b, 1); }

function render(fi) {
  const H = meta.hens;
  const pos = arrays.pos, heading = arrays.heading, headDown = arrays.head_down;
  const calls = arrays.calls, foodAmt = arrays.food_amount, struck = arrays.struck;
  const posS = stride('pos'), callS = stride('calls'), foodS = stride('food_amount');
  const now = performance.now() / 1000;

  for (let i = 0; i < H; i++) {
    const px = pos[fi * posS + i * 2], pz = pos[fi * posS + i * 2 + 1];
    const hd = heading[fi * H + i];
    dummy.position.set(px, HEN_PIVOT_Y, pz);
    dummy.rotation.set(0, 0, 0);
    dummy.lookAt(px + Math.cos(hd), HEN_PIVOT_Y, pz + Math.sin(hd));
    const down = headDown[fi * H + i];
    dummy.rotation.x = -down * 0.5;
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    henMesh.setMatrixAt(i, dummy.matrix);

    const s = struck[fi * H + i];
    if (lastStruck[i] === lastStruck[i] /* not NaN */ && s > lastStruck[i]) {
      flashUntil[i] = now + STRUCK_FLASH_S;
    }
    lastStruck[i] = s;
    const flashing = now < flashUntil[i];
    henMesh.setColorAt(i, flashing ? HEN_STRUCK : (down > 0.5 ? HEN_HEAD_DOWN : HEN_COLOR));

    // Calling indicator: a small sphere above the hen, coloured by call type,
    // visible only while amplitude clears the rest-floor.
    let bestCh = -1, bestAmp = CALL_ON;
    for (let c = 0; c < 4; c++) {
      const a = calls[fi * callS + i * 4 + c];
      if (a > bestAmp) { bestAmp = a; bestCh = c; }
    }
    if (bestCh >= 0) {
      dummy.position.set(px, 0.9, pz);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      henCallMesh.setColorAt(i, new THREE.Color(CALL_COLORS[bestCh]));
    } else {
      dummy.matrix.copy(HIDDEN);
    }
    henCallMesh.setMatrixAt(i, dummy.matrix);
  }
  henMesh.instanceMatrix.needsUpdate = true;
  henMesh.instanceColor.needsUpdate = true;
  henCallMesh.instanceMatrix.needsUpdate = true;
  henCallMesh.instanceColor.needsUpdate = true;

  for (let i = 0; i < meta.n_food; i++) {
    const amt = foodAmt[fi * foodS + i];
    const m4 = new THREE.Matrix4();
    foodMesh.getMatrixAt(i, m4);
    const p = new THREE.Vector3().setFromMatrixPosition(m4);
    dummy.position.copy(p);
    dummy.rotation.set(0, 0, 0);
    dummy.scale.set(1, Math.max(0.15, amt), 1);
    dummy.updateMatrix();
    foodMesh.setMatrixAt(i, dummy.matrix);
  }
  foodMesh.instanceMatrix.needsUpdate = true;

  const hawkOn = arrays.hawk_on[fi] > 0.5;
  hawkMesh.visible = hawkOn;
  if (hawkOn) hawkMesh.position.set(arrays.hawk_pos[fi * 2], 3.5, arrays.hawk_pos[fi * 2 + 1]);

  const foxOn = arrays.fox_on[fi] > 0.5;
  foxMesh.visible = foxOn;
  if (foxOn) foxMesh.position.set(arrays.fox_pos[fi * 2], 0.3, arrays.fox_pos[fi * 2 + 1]);

  const t = fi * meta.frame_dt, dur = (meta.n_frames - 1) * meta.frame_dt;
  clockEl.textContent = `${fmtClock(t)} / ${fmtClock(dur)}`;
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------
const clock = new THREE.Clock();
let bioT = 0;

function tick() {
  requestAnimationFrame(tick);
  const dt = clock.getDelta();
  controls.update();
  if (meta && playing) {
    const speed = parseFloat(speedSel.value);
    bioT += dt * speed;
    const dur = (meta.n_frames - 1) * meta.frame_dt;
    if (bioT > dur) bioT = 0; // loop -- this is for repeatedly watching a behaviour, not a one-shot player
    frameIdx = Math.min(meta.n_frames - 1, Math.floor(bioT / meta.frame_dt));
    scrub.value = String(frameIdx);
    render(frameIdx);
  }
  renderer.render(scene, camera);
}

playBtn.onclick = () => {
  playing = !playing;
  playBtn.textContent = playing ? 'pause' : 'play';
};
scrub.oninput = () => {
  playing = false;
  playBtn.textContent = 'play';
  frameIdx = parseInt(scrub.value, 10);
  bioT = frameIdx * meta.frame_dt;
  render(frameIdx);
};

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
(async () => {
  const runs = await fetchRuns();
  if (runs.length === 0) {
    emptyEl.style.display = 'flex';
    return;
  }
  for (const r of runs) {
    const opt = document.createElement('option');
    opt.value = r.id;
    opt.textContent = `${r.name} — ${r.hens}h/${r.minutes}m${r.plastic ? '/plastic' : ''}`;
    runSelect.appendChild(opt);
  }
  runSelect.onchange = () => { bioT = 0; playing = true; playBtn.textContent = 'pause'; loadRun(runSelect.value); };
  await loadRun(runs[0].id);
  tick();
})();
