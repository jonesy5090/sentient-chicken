// Offline replay viewer for sentient-chicken. Loads a recorded trajectory (flat
// float32 binary + JSON metadata, written by `run.record`) and plays it back as a
// stylised 3D scene. The sim itself is 2D (`World.pos` is `(H, 2)`) -- world x/y map
// to three.js x/z, "up" is added purely for legibility, not simulated.
import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';

const CALL_FLOOR = 0.0759;      // sigmoid(REST_BIAS); a silent hen still reads this
const CALL_ON = CALL_FLOOR + 0.15; // threshold above the floor to count as "calling"
// contact/food/aerial/ground/gakel (T2, E060) -- order must match spec.CALL_MOTOR_IDX.
const CALL_COLORS = [0xe0e0e0, 0x5ad86b, 0xff4d4d, 0xffb020, 0x9b59b6];
const STRUCK_FLASH_S = 0.4;
const SICK_FLASH_HZ = 2.5;   // blinks/sec for the sick-hen marker

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
let henMesh = null, henCallMesh = null, henSickMesh = null;
const dummy = new THREE.Object3D();
let flashUntil = null;   // per-hen wall-clock deadline for a struck flash
let lastStruck = null;   // per-hen cumulative strike-event count, previous frame

function buildHens(n) {
  if (henMesh) scene.remove(henMesh, henCallMesh, henSickMesh);
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

  // Sick marker: a separate, higher, larger sphere that blinks on/off, distinct from
  // the (brief, ~4s) gakel call indicator below it and from the muted body tint --
  // sickness lasts up to `sickness_duration_s` (60s default) and a hard on/off blink
  // reads as an alert in a way a static colour or a smooth pulse does not.
  henSickMesh = new THREE.InstancedMesh(
    new THREE.SphereGeometry(0.13, 8, 8),
    new THREE.MeshBasicMaterial({ color: 0xff3b3b, toneMapped: false }), n);
  scene.add(henSickMesh);

  flashUntil = new Float32Array(n).fill(-1);
  lastStruck = new Float32Array(n).fill(NaN);
}

const HEN_COLOR = new THREE.Color(0xd9c9a3);
const HEN_HEAD_DOWN = new THREE.Color(0x9c8a63);
const HEN_STRUCK = new THREE.Color(0xff2a2a);
// Visibly slow/still (T2, E060) -- a sickly yellow-green, distinct from head-down's
// plain dim tan so the two are never ambiguous at a glance.
const HEN_SICK = new THREE.Color(0x9fae3d);
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
let foodMesh = null, waterMesh = null, foodMarkerMesh = null;
const FOOD_SAFE = new THREE.Color(0x7a9a3d);
// Contaminated feeder (T2, E060) -- shown as ground truth for a human watching the
// replay, the same way hawk/fox position is always shown regardless of what a hen
// currently perceives. Real contamination stays invisible to the hens themselves
// (`coop/world.py`); this is a debug view, not a simulation of hen perception.
const FOOD_CONTAMINATED = new THREE.Color(0xb8433d);
// Fixed height for the top-face marker disc, comfortably above the tallest possible
// patch (cylinder height 0.15, centred at y=0.075, so a full patch's top face sits at
// 0.15) -- a flat marker rather than a colour alone, so contamination reads
// unambiguously even when the whole-patch tint is subtle at a distance or in fog.
const FOOD_MARKER_Y = 0.2;

function buildResources(foodPos, waterPos) {
  if (foodMesh) scene.remove(foodMesh, waterMesh, foodMarkerMesh);
  foodMesh = new THREE.InstancedMesh(
    new THREE.CylinderGeometry(0.35, 0.4, 0.15, 10),
    new THREE.MeshStandardMaterial({ vertexColors: false }), foodPos.length);
  foodMesh.receiveShadow = true;
  foodMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(foodPos.length * 3), 3);
  for (let i = 0; i < foodPos.length; i++) {
    dummy.position.set(foodPos[i][0], 0.075, foodPos[i][1]);
    dummy.rotation.set(0, 0, 0); dummy.scale.set(1, 1, 1); dummy.updateMatrix();
    foodMesh.setMatrixAt(i, dummy.matrix);
    foodMesh.setColorAt(i, FOOD_SAFE);
  }
  scene.add(foodMesh);

  foodMarkerMesh = new THREE.InstancedMesh(
    new THREE.CircleGeometry(0.14, 16),
    new THREE.MeshBasicMaterial({ color: 0xff3b3b, toneMapped: false, side: THREE.DoubleSide }),
    foodPos.length);
  for (let i = 0; i < foodPos.length; i++) {
    dummy.position.set(foodPos[i][0], FOOD_MARKER_Y, foodPos[i][1]);
    dummy.rotation.set(-Math.PI / 2, 0, 0);   // flat, facing up
    dummy.scale.set(1, 1, 1); dummy.updateMatrix();
    foodMarkerMesh.setMatrixAt(i, dummy.matrix);
  }
  scene.add(foodMarkerMesh);

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
const runNameEl = document.getElementById('runName');
const runMetaEl = document.getElementById('runMeta');
const runDescEl = document.getElementById('runDesc');
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
  runNameEl.textContent = m.name;
  runMetaEl.textContent = `${m.hens} hens · ${m.minutes} min · ${m.plastic ? 'plastic' : 'innate'}`;
  runDescEl.textContent = m.description || '(no description recorded for this run)';

  buildFloor(m.coop_size);
  buildHens(m.hens);
  buildResources(m.food_pos, m.water_pos);

  scrub.max = String(m.n_frames - 1);
  frameIdx = 0;
  scrub.value = '0';
  render(0);
}

function stride(name) { return meta.layout[name].shape.slice(1).reduce((a, b) => a * b, 1); }

// `sick_on`/`food_contaminated` only exist in trajectories recorded after E064 (T2's
// scaffold) -- an older recording, or anyone who hasn't re-recorded since, has no such
// field in `layout` at all. Missing entirely, not just empty, so this must be checked
// before `stride()` even runs (`meta.layout[name].shape` throws on `undefined`) --
// crashing here previously took the whole render loop down silently: `loadRun` calls
// `render(0)` synchronously, so the exception fired before `tick()` ever started,
// leaving the canvas blank forever while the sidebar (populated earlier in `loadRun`)
// looked fine. Missing data degrades to "nothing is sick/contaminated" instead.
function hasField(name) { return name in meta.layout; }

function render(fi) {
  const H = meta.hens;
  const pos = arrays.pos, heading = arrays.heading, headDown = arrays.head_down;
  const calls = arrays.calls, foodAmt = arrays.food_amount, struck = arrays.struck;
  const posS = stride('pos'), callS = stride('calls'), foodS = stride('food_amount');
  const now = performance.now() / 1000;

  const hasSick = hasField('sick_on');
  const hasContaminated = hasField('food_contaminated');
  const contS = hasContaminated ? stride('food_contaminated') : 0;

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
    const sick = hasSick && arrays.sick_on[fi * H + i] > 0.5;
    henMesh.setColorAt(i, flashing ? HEN_STRUCK :
      (sick ? HEN_SICK : (down > 0.5 ? HEN_HEAD_DOWN : HEN_COLOR)));

    // Calling indicator: a small sphere beside the hen, coloured by call type,
    // visible only while amplitude clears the rest-floor. Offset to the side (not
    // directly above her, where the sick marker below also sits) so a gakel call at
    // sickness onset -- the two markers' one guaranteed moment of overlap -- doesn't
    // have the smaller, non-flashing call dot visually swallowed by the bigger,
    // flashing sick marker sitting in the same column.
    let bestCh = -1, bestAmp = CALL_ON;
    for (let c = 0; c < CALL_COLORS.length; c++) {
      const a = calls[fi * callS + i * CALL_COLORS.length + c];
      if (a > bestAmp) { bestAmp = a; bestCh = c; }
    }
    if (bestCh >= 0) {
      dummy.position.set(px + 0.3, 0.9, pz);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      henCallMesh.setColorAt(i, new THREE.Color(CALL_COLORS[bestCh]));
    } else {
      dummy.matrix.copy(HIDDEN);
    }
    henCallMesh.setMatrixAt(i, dummy.matrix);

    // Sick flag: directly above her (not offset, unlike the call indicator above --
    // this is the more important of the two signals during a sickness event and
    // gets the more prominent central position), hard on/off blink (not a smooth
    // pulse) so it reads as an alert across the whole sickness window, not just the
    // brief gakel-call pulse at onset.
    if (sick && Math.floor(now * SICK_FLASH_HZ) % 2 === 0) {
      dummy.position.set(px, 1.3, pz);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
    } else {
      dummy.matrix.copy(HIDDEN);
    }
    henSickMesh.setMatrixAt(i, dummy.matrix);
  }
  henMesh.instanceMatrix.needsUpdate = true;
  henMesh.instanceColor.needsUpdate = true;
  henCallMesh.instanceMatrix.needsUpdate = true;
  henCallMesh.instanceColor.needsUpdate = true;
  henSickMesh.instanceMatrix.needsUpdate = true;

  const contaminated = arrays.food_contaminated;
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
    const isBad = hasContaminated && contaminated[fi * contS + i] > 0.5;
    foodMesh.setColorAt(i, isBad ? FOOD_CONTAMINATED : FOOD_SAFE);

    dummy.position.set(p.x, FOOD_MARKER_Y, p.z);
    dummy.rotation.set(-Math.PI / 2, 0, 0);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    foodMarkerMesh.setMatrixAt(i, isBad ? dummy.matrix : HIDDEN);
  }
  foodMesh.instanceMatrix.needsUpdate = true;
  foodMesh.instanceColor.needsUpdate = true;
  foodMarkerMesh.instanceMatrix.needsUpdate = true;

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
// An uncaught error anywhere in here previously left a blank canvas with no visible
// signal why: the scene, camera and lights are all built, but `renderer.render()`
// only ever runs inside `tick()`, and `tick()` only starts after `loadRun` resolves --
// so any exception before that point (a missing field, a bad fetch, ...) produces
// exactly "sidebar fine, scene never draws" with nothing in the page itself to say so.
(async () => {
  try {
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
    runSelect.onchange = () => {
      bioT = 0; playing = true; playBtn.textContent = 'pause';
      loadRun(runSelect.value).catch(reportBootError);
    };
    await loadRun(runs[0].id);
    tick();
  } catch (err) {
    reportBootError(err);
  }
})();

function reportBootError(err) {
  console.error('sentient-chicken viewer failed to load:', err);
  emptyEl.innerHTML = `<div>Failed to load this run.<br><code>${err.message}</code><br>`
    + `Check the browser console for details.</div>`;
  emptyEl.style.display = 'flex';
}
