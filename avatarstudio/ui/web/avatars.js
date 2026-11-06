// Avatars that aren't VRM files, for the 3D viewer (index.html loads this module when one is opened): glTF / GLB,
// FBX and MMD (PMX / PMD) humanoids, from one file, several files, a dropped folder or a .zip.
// The skeleton's bones are found by their names (Mixamo, Ready Player Me, VRoid, Unity, Unreal, Blender, 3ds Max,
// MMD's Japanese names). The model is turned to face like a VRM 1.0, scaled to metres and put in a T-pose (MMD
// models and some others stand in an A-pose); then the studio's pose stream drives it the way three-vrm drives a
// VRM's normalized bones. Faces move through blendshapes named after ARKit, VRoid, VRChat, Ready Player Me or MMD.
// Live2D models are recognised, not shown: they run in VTube Studio.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const VIRTUAL = 'https://avatar.files/';  // a model's own files seem to be here; the loaders' requests are mapped to them
const MODEL_TYPES = ['.vrm', '.pmx', '.pmd', '.glb', '.gltf', '.fbx'];  // (preferred in this order)
const LIVE2D = /\.(model3\.json|moc3|model\.json|moc)$/i;

// ---------------------------------------------------------------- the files
// path inside the model (folders with /) -> File or Blob. From a file picker (files), or a drop (entries: folders
// too); a .zip is replaced by what's in it.
export async function gather(files = [], entries = []) {
  const out = new Map();
  const add = (path, blob) => out.set(path.replace(/\\/g, '/').replace(/^\/+/, ''), blob);
  if (entries.length) for (const e of entries) await walk(e, '', add);
  else for (const f of files) add(f.webkitRelativePath || f.name, f);
  for (const [path, blob] of [...out]) {
    if (!/\.zip$/i.test(path)) continue;
    out.delete(path);
    const { unzipSync } = await import('three/addons/libs/fflate.module.js');
    for (const [name, data] of Object.entries(unzipSync(new Uint8Array(await blob.arrayBuffer())))) {
      if (!name.endsWith('/')) add(zipName(name), new Blob([data]));
    }
  }
  return out;
}

async function walk(entry, prefix, add) {
  if (entry.isFile) {
    add(prefix + entry.name, await new Promise((ok, bad) => entry.file(ok, bad)));
    return;
  }
  const reader = entry.createReader();
  for (;;) {  // (a folder comes in batches)
    const batch = await new Promise((ok, bad) => reader.readEntries(ok, bad));
    if (!batch.length) return;
    for (const e of batch) await walk(e, `${prefix}${entry.name}/`, add);
  }
}

// Zips made on Japanese Windows (most MMD models) store their file names in Shift-JIS, which the unzipper reads as
// Latin-1: back to bytes and decoded properly, or the model's texture names wouldn't match.
function zipName(name) {
  if (!/[\x80-\xff]/.test(name) || /[^\x00-\xff]/.test(name)) return name;
  const bytes = Uint8Array.from(name, (c) => c.charCodeAt(0));
  for (const enc of ['utf-8', 'shift_jis']) {
    try { return new TextDecoder(enc, { fatal: true }).decode(bytes); } catch (e) { /* the next one */ }
  }
  return name;
}

// What the files are: { kind: 'vrm' | 'model' | 'live2d' | 'none', main: the model file's path }
export function classify(files) {
  const paths = [...files.keys()], depth = (p) => p.split('/').length;
  for (const ext of MODEL_TYPES) {
    const found = paths.filter((p) => p.toLowerCase().endsWith(ext)).sort((a, b) => depth(a) - depth(b));
    if (found.length) return { kind: ext === '.vrm' ? 'vrm' : 'model', main: found[0] };
  }
  return { kind: paths.some((p) => LIVE2D.test(p)) ? 'live2d' : 'none' };
}

// The loaders ask for their textures by URL: those under VIRTUAL become the dropped files (by path, else by file
// name: FBX files often name their textures by a path on the artist's PC). What's missing is noted.
function resolver(files) {
  const byPath = new Map(), byName = new Map(), urls = new Map(), missing = new Set();
  for (const path of files.keys()) {
    byPath.set(path.toLowerCase(), path);
    const name = path.toLowerCase().split('/').pop();
    if (!byName.has(name)) byName.set(name, path);
  }
  const manager = new THREE.LoadingManager();
  manager.setURLModifier((url) => {
    if (!url.startsWith(VIRTUAL)) return url;
    let rel = url.slice(VIRTUAL.length).split(/[?#]/)[0];
    try { rel = decodeURIComponent(rel); } catch (e) { /* as it is */ }
    const parts = [];
    for (const p of rel.replace(/\\/g, '/').split('/')) {
      if (p === '..') parts.pop();
      else if (p && p !== '.') parts.push(p);
    }
    const key = parts.join('/').toLowerCase();
    const path = byPath.get(key) || byName.get(key.split('/').pop());
    if (!path) { missing.add(parts.join('/')); return url; }
    if (!urls.has(path)) urls.set(path, URL.createObjectURL(files.get(path)));
    return urls.get(path);
  });
  return {
    manager, missing,
    url: (path) => VIRTUAL + path.split('/').map(encodeURIComponent).join('/'),
    dispose() { for (const u of urls.values()) URL.revokeObjectURL(u); },
  };
}

async function loadScene(files, main, res, opts) {
  const ext = main.slice(main.lastIndexOf('.')).toLowerCase(), url = res.url(main);
  if (ext === '.fbx') {
    const { FBXLoader } = await import('three/addons/loaders/FBXLoader.js');
    return { root: await new FBXLoader(res.manager).loadAsync(url) };
  }
  if (ext === '.pmx' || ext === '.pmd') {
    const { MMDLoader } = await import('three/addons/loaders/MMDLoader.js');
    const mesh = await new MMDLoader(res.manager).loadAsync(url);
    return { root: mesh, mmd: mesh };
  }
  const [{ DRACOLoader }, { MeshoptDecoder }] = await Promise.all([  // (compressed glTF files)
    import('three/addons/loaders/DRACOLoader.js'), import('three/addons/libs/meshopt_decoder.module.js')]);
  const draco = new DRACOLoader().setDecoderPath(opts.dracoPath);
  const loader = new GLTFLoader(res.manager).setDRACOLoader(draco).setMeshoptDecoder(MeshoptDecoder);
  try {
    return { root: (await loader.loadAsync(url)).scene };
  } finally {
    draco.dispose();
  }
}

// ---------------------------------------------------------------- bones by name
const FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'];
const SEGMENTS = ['Proximal', 'Intermediate', 'Distal'];
// the studio's skeleton (avatar/rig.py REST): bone -> parent, parents first
const PARENT = { Hips: null, Spine: 'Hips', Chest: 'Spine', Neck: 'Chest', Head: 'Neck' };
for (const S of ['Left', 'Right']) {
  Object.assign(PARENT, {
    [`${S}Shoulder`]: 'Chest', [`${S}UpperArm`]: `${S}Shoulder`, [`${S}LowerArm`]: `${S}UpperArm`,
    [`${S}Hand`]: `${S}LowerArm`, [`${S}UpperLeg`]: 'Hips', [`${S}LowerLeg`]: `${S}UpperLeg`,
    [`${S}Foot`]: `${S}LowerLeg`, [`${S}Eye`]: 'Head' });
  for (const f of FINGERS) SEGMENTS.forEach((s, i) => { PARENT[`${S}${f}${s}`] = i ? `${S}${f}${SEGMENTS[i - 1]}` : `${S}Hand`; });
}
const ORDER = Object.keys(PARENT);
const CENTRE = new Set(['Hips', 'Spine', 'Chest', 'Neck', 'Head']);
// the studio's bones (without Left / Right) -> what models call them (lower case, letters and digits only), best first
const ALIASES = {
  Hips: ['hips', 'hip', 'pelvis'], Spine: ['spine', 'spine01', 'abdomen', 'spinelower', 'waist'],
  Chest: ['chest', 'spine1', 'spine02', 'spine2', 'spineupper'], Neck: ['neck', 'neck01', 'neck1'], Head: ['head'],
  Eye: ['eye', 'faceeye'], Shoulder: ['shoulder', 'clavicle', 'collar', 'collarbone'],
  UpperArm: ['upperarm', 'arm', 'uparm'], LowerArm: ['lowerarm', 'forearm', 'elbow'], Hand: ['hand', 'wrist'],
  UpperLeg: ['upperleg', 'upleg', 'thigh'], LowerLeg: ['lowerleg', 'leg', 'shin', 'calf', 'knee'],
  Foot: ['foot', 'ankle'],
};
const FINGER_WORDS = { thumb: 'Thumb', index: 'Index', middle: 'Middle', ring: 'Ring', little: 'Little', pinky: 'Little' };
// MMD: name -> [bone, rank]. センター before 腰: legs cancel 腰's turn (腰キャンセル), so it can't stand in for the hips
const MMD = { 'センター': ['Hips', 0], '腰': ['Hips', 1], '上半身': ['Spine', 0], '上半身2': ['Chest', 0], '首': ['Neck', 0],
  '頭': ['Head', 0] };
for (const [S, j] of [['Left', '左'], ['Right', '右']]) {
  for (const [jp, bone] of [['目', 'Eye'], ['肩', 'Shoulder'], ['腕', 'UpperArm'], ['ひじ', 'LowerArm'], ['肘', 'LowerArm'],
    ['手首', 'Hand'], ['足', 'UpperLeg'], ['ひざ', 'LowerLeg'], ['膝', 'LowerLeg'], ['足首', 'Foot']]) MMD[j + jp] = [S + bone, 0];
}
const MMD_FINGERS = { '親指': 'Thumb', '人指': 'Index', '人差指': 'Index', '中指': 'Middle', '薬指': 'Ring', '小指': 'Little' };
const halfWidth = (s) => s.replace(/[！-～]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0xFEE0));

// one bone's name -> { bone, rank } or { finger: 'LeftIndex', seg: 1 | 'proximal' ... } or null
function parseName(name) {
  const n = halfWidth(name.split(/[:|]/).pop());
  if (MMD[n]) return { bone: MMD[n][0], rank: MMD[n][1] };
  const jf = n.match(/^(左|右)(親指|人指|人差指|中指|薬指|小指)(\d)$/);
  if (jf) return { finger: (jf[1] === '左' ? 'Left' : 'Right') + MMD_FINGERS[jf[2]], seg: +jf[3] };
  let s = n.toLowerCase()
    .replace(/^(mixamorig\d*[_:.]?|j_bip_|j_adj_|bip0*1[\s_-]|def[-_.]|cc_base_|valvebiped\.bip01_|\d*joint[_-])/, '')
    .replace(/^c[_.]/, '');
  if (/(end|nub|tip|top)$/.test(s)) return null;  // (the ends of chains, not joints)
  let side = null;
  const m = s.match(/^(left|right)[_.\s-]?/) || s.match(/^([lr])[_.\s-]/) || s.match(/[_.\s-](left|right|[lr])$/)
    || s.match(/(left|right)$/);
  if (m) {
    side = m[1][0] === 'l' ? 'Left' : 'Right';
    s = s.slice(0, m.index) + s.slice(m.index + m[0].length);
  }
  const base = s.replace(/[^a-z0-9]/g, '');
  const f = base.match(/^(?:hand|f)?(thumb|index|middle|ring|little|pinky)(metacarpal|proximal|intermediate|distal|\d{1,2})$/);
  if (f) return side && { finger: side + FINGER_WORDS[f[1]], seg: /^\d/.test(f[2]) ? +f[2] : f[2] };
  const biped = base.match(/^finger([0-4])(\d?)$/);  // 3ds Max Biped: Finger0, Finger01, Finger02 (thumb) ... Finger42
  if (biped) return side && { finger: side + FINGERS[+biped[1]], seg: biped[2] === '' ? 0 : +biped[2] };
  for (const [bone, names] of Object.entries(ALIASES)) {
    const rank = names.indexOf(base);
    if (rank >= 0 && CENTRE.has(bone) === !side) return { bone: (side || '') + bone, rank };
  }
  return null;
}

// the model's bones for the studio's: { Hips: Bone, LeftUpperArm: Bone, ... }
function mapBones(root) {
  const bones = [];
  root.traverse((o) => { if (o.isBone) bones.push(o); });
  const best = {}, fingers = {};
  for (const bone of bones) {
    const p = parseName(bone.name || '');
    if (!p) continue;
    if (p.finger) (fingers[p.finger] ||= []).push({ bone, seg: p.seg });
    else if (!best[p.bone] || p.rank < best[p.bone].rank) best[p.bone] = { bone, rank: p.rank };
  }
  const map = Object.fromEntries(Object.entries(best).map(([k, v]) => [k, v.bone]));
  // A finger with 4 numbered bones: Mixamo's are 1-3 and an end (Index4), Biped's a metacarpal and 11-13. A
  // metacarpal starts at the wrist and is longer than the way to it; Mixamo's Index1 starts at the knuckle. (A
  // thumb's first bone always starts at the wrist: 4 thumb bones are 3 and an end.)
  root.updateMatrixWorld(true);
  const pos = (b) => b.getWorldPosition(new THREE.Vector3());
  const metacarpal = (key, hand, list) => !key.endsWith('Thumb') && hand
    && pos(list[0].bone).distanceTo(pos(hand)) < pos(list[1].bone).distanceTo(pos(list[0].bone));
  for (const [key, list] of Object.entries(fingers)) {
    const word = Object.fromEntries(list.filter((x) => typeof x.seg === 'string').map((x) => [x.seg, x.bone]));
    let chain;
    if (word.proximal || word.intermediate || word.distal) {  // Unity names the thumb's first bone Proximal, VRM 1.0 Metacarpal
      chain = key.endsWith('Thumb') && word.metacarpal ? [word.metacarpal, word.proximal, word.distal]
        : [word.proximal, word.intermediate, word.distal];
    } else {  // numbered: 1-3, 0-2 (a thumb's metacarpal), or 1-4 with an end bone
      let nums = [...new Map(list.filter((x) => typeof x.seg === 'number').map((x) => [x.seg, x])).values()]
        .sort((a, b) => a.seg - b.seg);
      if (nums.length >= 4) {
        nums = metacarpal(key, map[key.replace(/(Thumb|Index|Middle|Ring|Little)$/, 'Hand')], nums) ? nums.slice(1) : nums.slice(0, -1);
      }
      chain = nums.slice(0, 3).map((x) => x.bone);
    }
    chain.forEach((b, i) => { if (b) map[key + SEGMENTS[i]] = b; });
  }
  return map;
}

// ---------------------------------------------------------------- faces
const shapeKey = (n) => halfWidth(n.split(/[.|]/).pop()).toLowerCase().replace(/[\s_-]/g, '');
const SHAPES = {  // the studio's expressions (the stream's "vrm" values) -> blendshape names models use
  A: ['a', 'aa', 'v_aa', 'viseme_aa', 'Fcl_MTH_A', 'MTH_A', 'mouth_a', 'あ'],
  I: ['i', 'ih', 'v_ih', 'viseme_I', 'Fcl_MTH_I', 'MTH_I', 'mouth_i', 'い'],
  U: ['u', 'ou', 'v_ou', 'viseme_U', 'Fcl_MTH_U', 'MTH_U', 'mouth_u', 'う'],
  E: ['e', 'ee', 'v_e', 'viseme_E', 'Fcl_MTH_E', 'MTH_E', 'mouth_e', 'え'],
  O: ['o', 'oh', 'v_oh', 'viseme_O', 'Fcl_MTH_O', 'MTH_O', 'mouth_o', 'お'],
  Blink: ['blink', 'Fcl_EYE_Close', 'eyes_closed', 'まばたき'],
  // (MMD names its winks as seen from the front: ウィンク closes the eye on screen-left, the model's right)
  Blink_L: ['blink_l', 'blinkLeft', 'blink_left', 'Fcl_EYE_Close_L', 'wink_l', 'ウィンク右'],
  Blink_R: ['blink_r', 'blinkRight', 'blink_right', 'Fcl_EYE_Close_R', 'wink_r', 'ウィンク'],
  Joy: ['joy', 'happy', 'Fcl_ALL_Joy', 'smile', '笑い'], Angry: ['angry', 'Fcl_ALL_Angry', '怒り'],
  Sorrow: ['sorrow', 'sad', 'Fcl_ALL_Sorrow', '困る'], Fun: ['fun', 'relaxed', 'Fcl_ALL_Fun'],
  Surprised: ['surprised', 'Fcl_ALL_Surprised', 'びっくり'],
};
const ARKIT = /^(eye(blink|look|squint|wide)|jaw|mouth|brow|cheek|nose|tongue)/;

function faces(root) {
  const byKey = new Map();  // blendshape name (shapeKey) -> [[mesh, index], ...]
  root.traverse((o) => {
    if (!o.morphTargetDictionary || !o.morphTargetInfluences) return;
    for (const [name, i] of Object.entries(o.morphTargetDictionary)) {
      const k = shapeKey(name);
      if (!byKey.has(k)) byKey.set(k, []);
      byKey.get(k).push([o, i]);
    }
  });
  const presets = {};
  for (const [k, names] of Object.entries(SHAPES)) {
    const hit = names.map(shapeKey).find((n) => byKey.has(n));
    if (hit) presets[k] = byKey.get(hit);
  }
  if (presets.Blink_L && presets.Blink_R) delete presets.Blink;  // (the stream has both; one closing is enough)
  // Perfect Sync models (ARKit's 52 shapes) get the face from those; their vowel / blink shapes would add to it
  const perfectSync = [...byKey.keys()].filter((k) => ARKIT.test(k)).length >= 10;
  if (perfectSync) for (const k of ['A', 'I', 'U', 'E', 'O', 'Blink', 'Blink_L', 'Blink_R']) delete presets[k];
  return { byKey, presets, count: byKey.size };
}

// ---------------------------------------------------------------- loading and driving
// -> { object (add it to the scene), name, humanoid, bones, missing (files the model asked for), proportions,
//      pose(frame), update(k), dispose() }
export async function loadModel(files, main, opts) {
  const res = resolver(files);
  let scene;
  try {
    scene = await loadScene(files, main, res, opts);
  } catch (e) {
    res.dispose();
    throw e;
  }
  const { root, mmd } = scene;
  const map = mapBones(root);
  const humanoid = !!(map.Hips && (map.Head || map.Neck) && (map.LeftUpperArm || map.LeftUpperLeg));
  const holder = new THREE.Group(), align = new THREE.Group();  // holder: the avatar's own space, like a VRM 1.0's
  holder.add(align);
  align.add(root);
  const light = new THREE.DirectionalLight(0xffffff, 1.6);  // (these materials need a light from the front)
  light.position.set(0, 2, 3);
  holder.add(light);
  const at = (o) => { holder.updateMatrixWorld(true); return o.getWorldPosition(new THREE.Vector3()); };
  const Y = new THREE.Vector3(0, 1, 0);

  // stand it up (Z-up files), turn it so its left hand is at +X (facing +Z), scale to metres, feet on the floor
  if (humanoid) {
    const up = at(map.Head || map.Neck).sub(at(map.Hips));
    const axis = ['x', 'y', 'z'].reduce((a, k) => (Math.abs(up[k]) > Math.abs(up[a]) ? k : a), 'x');
    if (axis !== 'y' || up.y < 0) {
      const v = new THREE.Vector3();
      v[axis] = Math.sign(up[axis]);
      align.quaternion.setFromUnitVectors(v, Y);
    }
    const pair = [['LeftUpperArm', 'RightUpperArm'], ['LeftShoulder', 'RightShoulder'], ['LeftUpperLeg', 'RightUpperLeg']]
      .find(([a, b]) => map[a] && map[b]);
    if (pair) {
      const L = at(map[pair[0]]).sub(at(map[pair[1]]));
      align.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(Y, Math.atan2(L.z, L.x)));
    }
  }
  holder.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  const feet = ['LeftFoot', 'RightFoot'].filter((k) => map[k]).map((k) => at(map[k]).y);
  // its size in metres if it were a person: head to ankles ~1.37 m, hips to head ~0.6 m, else the whole thing ~1.6 m
  const ratio = !humanoid ? (box.max.y - box.min.y) / 1.6
    : feet.length ? (at(map.Head || map.Neck).y - Math.min(...feet)) / 1.37 : (at(map.Head || map.Neck).y - at(map.Hips).y) / 0.6;
  let s = ratio > 300 ? 0.001 : ratio > 30 ? 0.01 : ratio > 3 ? 0.08 : 1;  // millimetres, centimetres, MMD's units
  if (!(s * ratio > 0.4 && s * ratio < 2.5)) s = 1 / ratio;
  if (isFinite(s) && s > 0) align.scale.setScalar(s);
  holder.updateMatrixWorld(true);
  box.setFromObject(root);
  const centre = humanoid ? at(map.Hips) : box.getCenter(new THREE.Vector3());
  align.position.set(-centre.x, isFinite(box.min.y) ? -box.min.y : 0, -centre.z);

  // T-pose: arms in an A-pose (more than 15 degrees down) straight out to the sides, as the stream's rotations
  // assume. Everything else keeps the model's own rest, as a VRM's does: straightening the legs and hands too moved a
  // VRoid model's feet by 4 cm and fingertips by 1.5 cm compared with the same model loaded as a VRM.
  if (humanoid) {
    for (const [S, x] of [['Left', 1], ['Right', -1]]) {
      const OUT = new THREE.Vector3(x, 0, 0);
      for (const [a, b, dir] of [[`${S}UpperArm`, `${S}LowerArm`, OUT], [`${S}LowerArm`, `${S}Hand`, OUT]]) {
        if (!map[a] || !map[b]) continue;
        const from = at(map[b]).sub(at(map[a])).normalize();
        if (from.angleTo(dir) < THREE.MathUtils.degToRad(15)) continue;
        const pw = map[a].parent.getWorldQuaternion(new THREE.Quaternion());
        const r = new THREE.Quaternion().setFromUnitVectors(from, dir);
        map[a].quaternion.premultiply(pw.clone().invert().multiply(r).multiply(pw));  // (turned in the avatar's space)
      }
    }
  }
  holder.updateMatrixWorld(true);

  // the rest pose each driven bone keeps: three-vrm's normalized bones, worked out for any skeleton.
  // raw local = parentRest^-1 * turn * parentRest * restLocal, where turn is the stream's rotation of this bone
  // relative to the nearest driven bone above it (the bones in between, e.g. twist bones, keep their rest pose)
  const nameOf = new Map(Object.entries(map).map(([k, b]) => [b, k]));
  const driven = ORDER.filter((k) => map[k]).map((name) => {
    const bone = map[name];
    let up = bone.parent;
    while (up && up !== holder && !nameOf.has(up)) up = up.parent;
    const pw = bone.parent.getWorldQuaternion(new THREE.Quaternion());
    return { name, bone, from: nameOf.get(up) || null, rest: bone.quaternion.clone(), pw, pwInv: pw.clone().invert() };
  });
  const hipsRest = humanoid ? at(map.Hips) : null;
  const hipsParentInv = humanoid ? map.Hips.parent.matrixWorld.clone().invert() : null;
  // its bone layout for the studio (fingertip contacts, no clipping): offsets from each bone's parent, studio axes
  const proportions = {};
  for (const [name, parent] of Object.entries(PARENT)) {
    if (!parent || !map[name] || !map[parent]) continue;
    const o = at(map[name]).sub(at(map[parent]));
    proportions[name] = [-o.x, o.y, o.z];
  }
  // MMD's "grant" bones copy part of another bone's turn (the legs' D bones, twist helpers)
  const grants = ((mmd && mmd.geometry.userData.MMD && mmd.geometry.userData.MMD.grants) || [])
    .filter((g) => !g.isLocal && g.affectRotation);
  const grantRest = grants.map((g) => mmd.skeleton.bones[g.index].quaternion.clone());
  holder.rotation.y = Math.PI;  // face this scene's camera, like the VRMs

  const face = faces(root);
  const target = new Map(), current = new Map(), W = {};
  for (const k of ORDER) W[k] = new THREE.Quaternion();
  const hipsTarget = new THREE.Vector3(), hipsNow = new THREE.Vector3(), q = new THREE.Quaternion(), v = new THREE.Vector3();
  let posed = false;
  const name = main.split('/').pop().replace(/\.[^.]+$/, '');

  return {
    object: holder, name, humanoid, bones: driven.length, missing: [...res.missing], proportions,
    shapes: face.count,
    pose(d) {  // a frame from the studio: targets (update() glides to them) and the face at once
      if (humanoid) {
        for (const [k, r] of Object.entries(d.bones || {})) {
          if (!(k in PARENT)) continue;
          const t = target.get(k) || new THREE.Quaternion();
          t.set(r[0], -r[1], -r[2], r[3]);  // (the stream in VRM 1.0's axes, as index.html's driveVRM)
          if (!target.has(k)) { target.set(k, t); current.set(k, t.clone()); }
        }
        const r = d.root || [0, 0, 0];
        hipsTarget.set(-r[0], r[1], r[2]);
        if (!posed) hipsNow.copy(hipsTarget);
        posed = true;
      }
      for (const [k, list] of Object.entries(face.presets)) {
        const value = (d.vrm && d.vrm[k]) || 0;
        for (const [m, i] of list) m.morphTargetInfluences[i] = value;
      }
      for (const [k, value] of Object.entries(d.arkit || {})) {
        for (const [m, i] of face.byKey.get(shapeKey(k)) || []) m.morphTargetInfluences[i] = value;
      }
    },
    update(k) {  // each drawn frame: k = how far to glide toward the latest pose (0..1)
      if (!humanoid || !posed) return;
      for (const [name, t] of target) current.get(name).slerp(t, k);
      hipsNow.lerp(hipsTarget, k);
      for (const name of ORDER) {
        const p = PARENT[name];
        W[name].copy(p ? W[p] : q.identity());
        if (current.has(name)) W[name].multiply(current.get(name));
      }
      for (const b of driven) {
        if (b.from) q.copy(W[b.from]).invert().multiply(W[b.name]);
        else q.copy(W[b.name]);
        b.bone.quaternion.copy(b.pwInv).multiply(q).multiply(b.pw).multiply(b.rest);
      }
      map.Hips.position.copy(v.copy(hipsRest).add(hipsNow).applyMatrix4(hipsParentInv));
      grants.forEach((g, i) => {
        const bones = mmd.skeleton.bones;
        bones[g.index].quaternion.copy(grantRest[i]).multiply(q.identity().slerp(bones[g.parentIndex].quaternion, g.ratio));
      });
    },
    dispose() {
      holder.removeFromParent();
      holder.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        for (const m of [].concat(o.material || [])) {
          for (const value of Object.values(m)) if (value && value.isTexture) value.dispose();
          m.dispose();
        }
        if (o.skeleton) o.skeleton.dispose();
      });
      res.dispose();
    },
    debug: { map, driven, align },  // (for automated checks)
  };
}
