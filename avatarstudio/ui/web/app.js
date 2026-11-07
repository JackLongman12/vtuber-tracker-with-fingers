// Avatar Studio app window (app.html). Two connections:
//  - this window's own Python side (mainwindow.AppApi, through pywebview): start / stop the engine and the tools,
//    devices, settings while tracking is off;
//  - the engine (studio.py --ui) over its WebSocket, like the old settings page: status, prompts, the eye-gaze dot,
//    settings changes and the studio keys. Its camera picture comes as camera.mjpg; the avatar pane is the 3D viewer
//    page (index.html?embed) in a frame.
'use strict';
const $ = (s, el = document) => el.querySelector(s);
// a link or file dropped onto the window would open in it (and get its controls): nothing here takes drops
// (the 3D avatar pane has its own VRM drop)
addEventListener('dragover', (e) => e.preventDefault());
addEventListener('drop', (e) => e.preventDefault());
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const WS_URL = 'ws://127.0.0.1:8765', CAMERA_URL = 'http://127.0.0.1:8000/camera.mjpg';
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const icon = (name) => `<svg class="i"><use href="#i-${name}"/></svg>`;
// tools that use the camera run like the engine, on the Live page (their picture, prompts, Cancel)
const TOOL_MODES = { train_face: 'Face training', gesture: 'Recording a gesture' };
const ENGINE_LOGS = { train_face: 'train_face', gesture: 'gestures' };  // (where their messages show afterwards)
const JOB_LOGS = { 'train-gestures': 'gestures' };

const app = {
  engine: null, jobs: {}, prefs: {}, devices: null, platform: 'win32', tray: false,
  status: null, settings: null, profiles: [], hotkeys: {},
  page: 'live', layout: 'split', ws: null, wsOpen: false, editing: 0, seenLines: {}, toldExit: null,
  stopping: false, autostarted: false, filled: false,
};

// ---------------------------------------------------------------- this window's Python side
let apiResolve;
const apiReady = new Promise((r) => { apiResolve = r; });
addEventListener('pywebviewready', () => apiResolve(window.pywebview.api));
if (window.pywebview && window.pywebview.api) apiResolve(window.pywebview.api);
async function call(name, ...args) {
  const api = await apiReady;
  return api[name](...args);
}

// ---------------------------------------------------------------- the engine's WebSocket
let retry = 400, retryTimer = null;
function connect() {
  clearTimeout(retryTimer);
  if (app.ws && app.ws.readyState <= 1) return;  // (open or opening already)
  const ws = new WebSocket(WS_URL);
  app.ws = ws;
  ws.onopen = () => { app.wsOpen = true; retry = 400; render(); };
  ws.onclose = () => {
    if (app.ws !== ws) return;
    app.wsOpen = false; app.status = null; render();
    retryTimer = setTimeout(connect, engineRunning() ? 400 : retry);
    retry = Math.min(retry * 1.6, 4000);
  };
  ws.onmessage = (e) => {
    const d = JSON.parse(e.data);
    if (d.type === 'frame') onFrame(d);
  };
}
function send(m) {
  if (app.ws && app.ws.readyState === 1) { app.ws.send(JSON.stringify(m)); return true; }
  return false;
}
function sendKey(k) {
  if (!send({ type: 'key', key: k })) toast('Start tracking first');
}
const engineRunning = () => !!(app.engine && app.engine.running);
const tracking = () => engineRunning() && app.wsOpen && !!app.status;

let statusShown = 0;
function onFrame(d) {
  if (d.status) {
    app.status = d.status;
    app.settings = { ...(app.settings || {}), ...d.status.settings };
    app.hotkeys = d.status.hotkeys || app.hotkeys;
    app.profiles = d.status.profiles || app.profiles;
    if (performance.now() - statusShown > 240) { statusShown = performance.now(); render(); }
  }
  if (d.prompt) showPrompt(d.prompt, d.progress);
  if (d.gaze_target) showGaze(d.gaze_target);
}

// ---------------------------------------------------------------- prompts and the eye-gaze dots
let promptTimer = null;
function showPrompt(lines, progress) {
  const el = $('#prompt'), key = lines.join('\n');
  if (el.dataset.key !== key) {
    el.dataset.key = key;
    el.innerHTML = lines.map((l, i) => i > 0 && /^#*$/.test(l.trim())
      ? `<div class="bar"><span style="width:${Math.min(100, l.trim().length * 5)}%"></span></div>`
      : `<div class="line${i === 0 ? ' first' : ''}">${esc(l.replace(/ {3,}/g, '  '))}</div>`).join('')
      + (progress != null ? '<div class="bar"><span data-progress></span></div>' : '');
  }
  const bar = $('[data-progress]', el);
  if (bar) bar.style.width = `${Math.round((progress || 0) * 100)}%`;
  el.classList.remove('hidden');
  clearTimeout(promptTimer);
  promptTimer = setTimeout(() => el.classList.add('hidden'), 450);
}
let gazeTimer = null, gazeFull = false;
function showGaze([tx, ty, i, n, dt]) {
  const el = $('#gaze'), dot = $('#gaze-dot');
  el.classList.remove('hidden');
  if (!gazeFull) { gazeFull = true; call('fullscreen', true); }
  dot.style.left = `${50 + tx * 42}%`;  // (where the studio's own full-screen window put them)
  dot.style.top = `${50 + ty * 40}%`;
  dot.classList.toggle('settling', dt < 1.0);
  $('#gaze-text').textContent = `Look at the dot with your eyes only and keep your head still  (${i + 1} of ${n})`;
  clearTimeout(gazeTimer);
  gazeTimer = setTimeout(() => { el.classList.add('hidden'); gazeFull = false; call('fullscreen', false); }, 600);
}

// ---------------------------------------------------------------- toasts
function toast(text, kind = '') {
  const t = document.createElement('div');
  t.className = `toast ${kind}`;
  t.textContent = text;
  $('#toasts').appendChild(t);
  setTimeout(() => t.remove(), kind === 'bad' ? 9000 : 4500);
}
// something stopped by itself: what stopped, its last message, and a bug report already filled in with it
const MODE_NAMES = { live: 'Tracking', latency: 'The latency test', test_pattern: 'The avatar test',
  train_face: 'Face training', gesture: 'Gesture recording' };
const TOOL_NAMES = { 'train-gestures': 'Gesture training', 'tune-smoothing': 'Smoothing tuning', mocap: 'Motion capture',
  'wilor-setup': 'The WiLoR setup' };
function crashNotice(what, lines = [], after = '', verb = 'stopped unexpectedly') {
  const last = [...lines].reverse().find((l) => l.trim()) || '';
  const t = document.createElement('div');
  t.className = 'toast bad crash';
  t.innerHTML = `<b>${esc(what)} ${esc(verb)}.</b>${after ? ` ${esc(after)}` : ''}`
    + (last ? `<div class="crash-line">${esc(last)}</div>` : '')
    + '<div class="crash-actions"><button class="btn primary" data-act="report">Report this problem</button><button class="btn ghost" data-act="close">Dismiss</button></div>';
  const close = () => t.parentNode && t.parentNode.removeChild(t);
  $('[data-act="close"]', t).onclick = close;
  $('[data-act="report"]', t).onclick = () => {
    close();
    openReport(`${what} ${verb}.\n\nLast messages:\n${lines.slice(-8).join('\n')}`);
  };
  const box = $('#toasts');
  box.insertBefore(t, box.firstChild);
  $$('.toast.crash', box).slice(3).forEach((old) => box.removeChild(old));  // (at most three at once)
}
const NOTABLE = /calibrated|calibration|saved|profile|failed|unavailable|busy|lost|could not|can't|reset to defaults|field of view set/i;
function friendly(line) {  // the engine's messages, minus file paths and lists of what was measured
  if (/^calibrated \(/.test(line)) return 'Neutral pose calibrated';
  if (/^video saved -> /.test(line)) return `Recording saved: ${line.split('-> ')[1]}`;
  const s = line.replace(/\s*->\s*\S.*$/, '').replace(/\s*\([^)]*\)/g, '').trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// ---------------------------------------------------------------- state from the Python side
async function poll() {
  try { applyState(await call('get_state')); } catch (e) { /* (the window is closing) */ }
  setTimeout(poll, 600);
}
function newLines(p) {
  if (!p) return [];
  const key = `${p.name}@${p.id}`;
  const seen = app.seenLines[key] || 0;
  app.seenLines[key] = p.count;
  return p.lines.slice(Math.max(0, p.lines.length - (p.count - seen)));
}
function applyState(s) {
  const wasRunning = engineRunning();
  app.engine = s.engine; app.jobs = s.jobs; app.prefs = s.prefs || {}; app.platform = s.platform; app.tray = s.tray;
  if (s.devices.cameras !== null || !app.devices) app.devices = s.devices;
  $('#version').textContent = `version ${s.version}${s.packaged ? '' : ' · development'}`;
  app.version = s.version; app.packaged = s.packaged; app.updatesFrom = s.updates_from;
  if (s.update.state === 'available' && app.update?.state !== 'available' && !app.toldUpdate) {
    app.toldUpdate = true;
    toast(`Avatar Studio ${s.update.info.version} is available: Update in the top bar`, 'ok');
  }
  app.update = s.update;
  renderUpdate();
  if (!app.filled) {
    app.filled = true;
    setLayout(app.prefs.layout || 'split', false);
    if (app.prefs.autostart && !engineRunning() && !app.autostarted) { app.autostarted = true; start('live'); }
    if (!app.prefs.guide_done) maybeShowGuide();
  }
  fillDevices();
  const e = s.engine;
  if (e) {
    for (const line of newLines(e)) {
      const part = /^(.+?) failed on a frame/.exec(line);  // (an error the engine carried on after)
      if (part) crashNotice(`${part[1].charAt(0).toUpperCase()}${part[1].slice(1)}`, e.lines, 'Tracking carries on.', 'had an error');
      else if (NOTABLE.test(line)) toast(friendly(line), /failed|unavailable|busy|lost|could not|can't/i.test(line) ? 'bad' : 'ok');
    }
    if (!e.running && e.code && !e.stopped && app.toldExit !== e.id) {  // (it ended by itself, with an error)
      app.toldExit = e.id;
      crashNotice(MODE_NAMES[e.mode] || 'Tracking', e.lines);
    }
    const wl = e.running && app.status ? app.status.wilor : null;  // (the finger helper ended; tracking goes on)
    if (wl === 'failed' && app.toldWilor !== e.id) {
      app.toldWilor = e.id;
      crashNotice('The WiLoR finger helper', e.lines.filter((l) => /wilor/i.test(l)), 'Fingers carry on with MediaPipe.');
    }
    if (e.mode === 'latency') {
      const result = e.lines.filter((l) => /clap test|motion -> avatar/i.test(l));
      if (result.length) { const el = $('#latency-result'); el.textContent = result.join('\n'); el.classList.remove('hidden'); }
    }
    if (e.running) $('#avatar').contentWindow?.postMessage('connect', '*');  // (the avatar connects at once)
    if (ENGINE_LOGS[e.mode]) showLog(ENGINE_LOGS[e.mode], e);
  }
  for (const [name, j] of Object.entries(s.jobs)) {
    showLog(JOB_LOGS[name] || name, j);
    showProgress(name, j);
    if (name === 'wilor-setup' && app.wilorSeen !== `${j.id}${j.running}`) { app.wilorSeen = `${j.id}${j.running}`; fillWilor(); }
    for (const line of newLines(j)) if (/saved|written|done|->/i.test(line) && !j.running) toast(friendly(line), 'ok');
    app.toldJobs = app.toldJobs || {};
    if (!j.running && j.code && !j.stopped && app.toldJobs[name] !== j.id) {
      app.toldJobs[name] = j.id;
      crashNotice(TOOL_NAMES[name] || name, j.lines);
    }
  }
  if (wasRunning && !engineRunning()) {
    app.status = null; app.stopping = false;
    if (app.returnPage) { showPage(app.returnPage); app.returnPage = null; }  // (a tool finished: back where it began)
  }
  const gjob = s.jobs['train-gestures'];
  if ((wasRunning && !engineRunning() && e && e.mode === 'gesture') || (gjob && !gjob.running && app.gestureJob !== gjob.id)) {
    if (gjob && !gjob.running) app.gestureJob = gjob.id;
    loadGestures();
  }
  render();
}
function showLog(where, p) {
  const log = $(`[data-log="${where}"]`);
  if (!log) return;
  log.classList.remove('hidden');
  log.textContent = (p.running ? '' : p.code ? `Ended with an error (${p.code}).\n` : 'Done.\n') + p.lines.join('\n');
  log.scrollTop = log.scrollHeight;
}
// progress of the background tools, from what they print
function showProgress(name, j) {
  const el = $(`[data-progress="${name}"]`);
  if (!el) return;
  const all = j.lines.join('\n');
  let pct = null, text = j.lines.slice(-1)[0] || 'Starting…';
  if (name === 'mocap') {
    const m = [...all.matchAll(/tracked (\d+)\/(\d+) frames/g)].pop();
    if (/done ->/.test(all)) pct = 1;
    else if (/drawing overlay/.test(all)) { pct = 0.92; text = 'Drawing the overlay video…'; }
    else if (/cleaning up/.test(all)) { pct = 0.85; text = 'Filtering the whole take…'; }
    else if (m) { pct = 0.8 * (+m[1]) / Math.max(+m[2], 1); text = `Tracking: ${m[1]} of ${m[2]} frames`; }
  } else if (name === 'wilor-setup') {
    const m = [...all.matchAll(/step (\d+)\/(\d+): ([^\n]*)/g)].pop();
    if (m) { pct = (+m[1] - 1) / +m[2]; text = `Step ${m[1]} of ${m[2]}: ${m[3]}`; }
  } else if (name === 'tune-smoothing') {
    if (/saved ->|already as good/.test(all)) pct = 1;
    else if (/tracked \d+ frames/.test(all)) { pct = 0.7; text = 'Comparing smoothing settings…'; }
    else text = 'Tracking your take…';
  }
  if (!j.running) {
    pct = j.code ? 0 : 1;
    text = j.code ? 'Ended with an error (see below)'
      : name === 'mocap' ? 'Done: BVH, CSV, face curves and the overlay video are in the results folder'
      : name === 'wilor-setup' ? 'Done: WiLoR is used from the next start of tracking'
        : /already as good/.test(all) ? 'Done: the defaults already suit your take' : 'Done: smoothing tuned to you';
  }
  el.classList.remove('hidden');
  el.classList.toggle('busy', j.running && pct === null);
  el.classList.toggle('done', !j.running && !j.code);
  $('.bar span', el).style.width = `${Math.round((pct || 0) * 100)}%`;
  $('.ptext', el).textContent = text;
}

// ---------------------------------------------------------------- start / stop
async function start(mode, arg) {
  const r = await call('start', mode, arg ?? null);
  if (r && !r.ok && r.error) { toast(r.error, 'bad'); return; }
  retry = 400; setTimeout(connect, 300);  // (connect() leaves an open connection alone)
  if (TOOL_MODES[mode] && app.page !== 'live') app.returnPage = app.page;
  if (mode !== 'live') showPage('live');
  if (mode === 'test_pattern') setLayout('avatar');
  applyState(await call('get_state'));
}
async function stop() {
  app.stopping = true; render();
  await call('stop');
  app.stopping = false;
  applyState(await call('get_state'));
}
$('#btn-track').onclick = () => (engineRunning() ? stop() : start('live'));
document.addEventListener('click', (e) => {
  const b = e.target.closest('[data-start]');
  if (b) start(b.dataset.start);
  const t = e.target.closest('[data-tool]');
  if (t) runTool(t.dataset.tool);
  const k = e.target.closest('[data-key]');
  if (k) sendKey(k.dataset.key);
});
async function runTool(name) {
  const r = await call('run_tool', name);
  if (r && !r.ok && r.error) toast(r.error, 'bad');
  applyState(await call('get_state'));
}
$('#btn-mocap-results').onclick = async () => { const r = await call('open_mocap_results'); if (!r.ok) toast(r.error); };
$('#btn-data').onclick = () => call('open_data_folder', '');

// ---------------------------------------------------------------- devices (top bar, settings)
function options(sel, items, value) {
  const html = items.map(([v, label]) => `<option value="${esc(v)}">${esc(label)}</option>`).join('');
  if (sel.dataset.html !== html) { sel.innerHTML = html; sel.dataset.html = html; }
  if (value !== undefined && items.some(([v]) => v === value)) sel.value = value;
}
function fillDevices() {
  const cams = app.devices && app.devices.cameras;
  const camItems = cams ? (cams.length ? cams : ['Camera 0']).map((c) => [c, c]) : [['', 'Searching for cameras…']];
  options($('#sel-camera'), camItems, app.prefs.camera || (cams && cams[0]));
  const def = 'Windows default';  // (the name the launcher stores for "the OS's default microphone", on any OS)
  options($('#sel-mic'), [[def, 'Default microphone'], ...(app.devices?.mics || []).map((m) => [m, m]),
    ['No microphone', 'No microphone']], app.prefs.mic || def);
  for (const key of ['camera2', 'camera3']) {
    const sel = $(`#sel-${key}`);
    if (sel) options(sel, [['none', 'None'], ...(cams || []).map((c) => [c, c])], app.prefs[key] || 'none');
  }
  const fps = $('#sel-fps');
  if (fps) options(fps, [['30 fps', '30 fps'], ['60 fps (if the camera can)', '60 fps, if the camera can']], app.prefs.fps || '30 fps');
}
async function setPref(values) {
  app.prefs = await call('set_prefs', values);
  const device = ['camera', 'mic', 'fps', 'camera2', 'camera3'].some((k) => k in values);
  if (device && engineRunning() && app.engine.mode === 'live') {  // (the engine opens its devices when it starts)
    toast('Restarting tracking with the new device…');
    await call('stop');
    start('live');
  }
}
$('#sel-camera').onchange = (e) => setPref({ camera: e.target.value });
$('#sel-mic').onchange = (e) => setPref({ mic: e.target.value });

// ---------------------------------------------------------------- pages and layout
function showPage(name) {
  app.page = name;
  $$('.nav-item').forEach((b) => b.classList.toggle('active', b.dataset.page === name));
  $$('.page').forEach((p) => p.classList.toggle('active', p.id === `page-${name}`));
  if (name === 'settings') loadSettings();
  if (name === 'start') loadChecklist();
  if (name === 'personalize') loadGestures();
  if (name === 'check') checkConnections();
  render();
}
$$('.nav-item').forEach((b) => (b.onclick = () => showPage(b.dataset.page)));
function setLayout(layout, save = true) {
  app.layout = layout;
  $('#stage').className = `stage ${layout}`;
  $$('#layout button').forEach((b) => b.classList.toggle('on', b.dataset.layout === layout));
  if (save) call('set_prefs', { layout });
  render();
}
$$('#layout button').forEach((b) => (b.onclick = () => setLayout(b.dataset.layout)));
$('#btn-vrm').onclick = () => {
  try { $('#avatar').contentDocument.getElementById('load-avatar').click(); } catch (e) { toast('The avatar view is still loading'); }
};
$$('#overlays .chip').forEach((c) => (c.onclick = () => changeSetting(c.dataset.setting, !(app.settings || {})[c.dataset.setting])));

// keys: the studio's own keys while this window (or the avatar inside it) is in front
const KEYS = 'cvkgpmbfh123456789';
function onKey(key, mods, target) {
  if (mods || key.length !== 1 || !KEYS.includes(key.toLowerCase())) return false;
  if (target && /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName)) return false;
  if (!$('#gaze').classList.contains('hidden') && key !== 'p') return true;
  const toggles = { m: 'show_mesh', b: 'show_body', f: 'show_face_monitor', h: 'show_panel' };
  const k = key.toLowerCase();
  if (toggles[k]) changeSetting(toggles[k], !(app.settings || {})[toggles[k]]);
  else sendKey(k);
  return true;
}
addEventListener('keydown', (e) => { if (onKey(e.key, e.ctrlKey || e.altKey || e.metaKey, e.target)) e.preventDefault(); });
addEventListener('message', (e) => {  // (keys pressed while the avatar frame has the focus)
  if (e.data && e.data.type === 'key') onKey(e.data.key, e.data.mods, null);
  if (e.data && e.data.type === 'avatar-note') toast(e.data.text);
  if (e.data && e.data.type === 'live2d') {  // (Live2D's runtime needs its own licence: those models run in VTube Studio)
    toast('Live2D models run in VTube Studio: load the model there, turn on its API (port 8001), then turn on VTube Studio (Live2D) in Settings here.');
  }
});

// ---------------------------------------------------------------- what the page shows
const camera = $('#camera');
let cameraWanted = false, cameraRetry = null, cameraCheck = null;
function setHTML(el, html) { if (el.dataset.html !== html) { el.innerHTML = html; el.dataset.html = html; } }
function wantCamera(on) {
  if (on === cameraWanted) return;
  cameraWanted = on;
  clearTimeout(cameraRetry);
  if (on) camera.src = `${CAMERA_URL}?t=${Date.now()}`;
  else camera.removeAttribute('src');  // (closes the stream: the engine stops drawing the picture)
}
camera.onerror = () => {
  if (!cameraWanted) return;
  cameraRetry = setTimeout(() => { if (cameraWanted) camera.src = `${CAMERA_URL}?t=${Date.now()}`; }, 700);
};
function render() {
  const run = engineRunning(), st = app.status, mode = app.engine && app.engine.mode;
  // top bar
  const pill = $('#engine-pill');
  let text = 'Not tracking', cls = '';
  if (app.stopping) { text = 'Stopping…'; cls = 'warn'; }
  else if (run && mode === 'test_pattern') { text = 'Avatar test'; cls = 'info'; }
  else if (run && (!app.wsOpen || !st)) { text = 'Starting…'; cls = 'warn'; }
  else if (run && TOOL_MODES[mode]) { text = TOOL_MODES[mode]; cls = 'info'; }
  else if (run && mode.startsWith('record')) { text = `Recording · ${fmtTime(app.engine.since)}`; cls = 'bad'; }
  else if (run && mode === 'latency') { text = 'Latency test'; cls = 'info'; }
  else if (run && st.paused) { text = 'Paused'; cls = 'warn'; }
  else if (run) { text = st.fps >= 1 ? `Tracking · ${Math.round(st.fps)} fps` : 'Tracking'; cls = 'ok'; }
  pill.textContent = text; pill.className = `pill ${cls}`;
  const btn = $('#btn-track');
  btn.className = run ? 'btn big' : 'btn primary big';
  setHTML(btn, run ? `${icon('stop')}<span>${mode && mode.startsWith('record') ? 'Stop recording' : TOOL_MODES[mode] ? 'Cancel' : 'Stop'}</span>`
    : `${icon('play')}<span>Start tracking</span>`);
  btn.disabled = app.stopping;
  $('#page-live').classList.toggle('tool-mode', run && !!TOOL_MODES[mode]);
  // live page
  const live = app.page === 'live';
  const wantPicture = live && run && mode !== 'test_pattern' && app.layout !== 'avatar' && !app.stopping;
  wantCamera(wantPicture);
  const empty = $('#camera-empty'), showing = wantPicture && camera.naturalWidth > 0;
  empty.classList.toggle('hidden', showing);
  camera.classList.toggle('hidden', !showing);
  let avatarName = '';
  try { avatarName = $('#avatar').contentDocument.getElementById('avatar-name').textContent; } catch (e) { /* (loading) */ }
  const vrm = avatarName && avatarName !== 'skeleton';
  $('#avatar-empty').classList.toggle('hidden', vrm || (run && app.wsOpen));
  $('#avatar-label').textContent = vrm ? `Avatar · ${avatarName}` : 'Avatar · drop your model here';
  if (!showing) {
    const [title, body] = !run ? ['Your camera shows here', 'Start tracking to see what the studio sees.']
      : mode === 'test_pattern' ? ['The avatar test doesn’t use the camera', 'Watch the avatar and your VTuber app.']
        : ['Starting the camera…', 'This takes a moment.'];
    $('#camera-empty-title').textContent = title; $('#camera-empty-text').textContent = body;
    $('#camera-empty .btn').classList.toggle('hidden', run);
  }
  if (live && camera.naturalWidth === 0 && wantPicture && !cameraCheck) {  // (until the first picture arrives)
    cameraCheck = setTimeout(() => { cameraCheck = null; render(); }, 300);
  }
  $$('#overlays .chip').forEach((c) => c.classList.toggle('on', !!(app.settings || {})[c.dataset.setting]));
  const pause = $('#btn-pause');
  pause.classList.toggle('on', !!(st && st.paused));
  pause.querySelector('span').textContent = st && st.paused ? 'Resume' : 'Pause';
  renderExpressions();
  renderStatusbar();
  if (app.page === 'settings' && performance.now() - app.editing > 700) fillSettings();
}
function fmtTime(s) { s = Math.max(0, Math.round(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; }
function renderExpressions() {
  const box = $('#expressions'), keys = (app.status && app.status.hotkeys) || app.hotkeys || {};
  const sig = JSON.stringify(keys);
  if (box.dataset.sig !== sig) {
    box.dataset.sig = sig;
    box.innerHTML = '<span class="group-label">Expressions</span>' + Object.entries(keys).map(([k, n]) =>
      `<button class="chip" data-expr="${esc(n)}" title="Key ${esc(k)} (Ctrl+Alt+${esc(k)} from any window)">${esc(k)} · ${esc(n)}</button>`).join('');
    $$('[data-expr]', box).forEach((b) => (b.onclick = () => {
      if (!send({ type: 'action', name: 'toggle_expression', arg: b.dataset.expr })) toast('Start tracking first');
    }));
  }
  const active = (app.status && app.status.active) || [];
  $$('[data-expr]', box).forEach((b) => b.classList.toggle('on', active.includes(b.dataset.expr)));
}
function renderStatusbar() {
  const st = app.status;
  const set = (id, text) => { $(id).textContent = text; };
  const known = (k) => st && st[k] !== undefined;  // (the tools report only some of these)
  set('#st-face', known('face') ? (st.covered ? 'camera covered' : st.face ? 'tracked' : 'not visible') : '–');
  $('#st-face-dot').className = `dot ${known('face') ? (st.face ? 'ok' : 'warn') : ''}`;
  set('#st-hands', known('hands') ? String(st.hands) : '–');
  set('#st-cal', known('calibrated') ? (st.calibrated ? (st.vowels ? 'neutral + vowels' : 'neutral only') : 'not calibrated') : '–');
  $('#st-mic').style.width = `${Math.round(((st && st.mic) || 0) * 100)}%`;
  const wl = known('wilor') ? st.wilor || 'off' : null;
  set('#st-wilor', !wl ? '–' : wl === 'running' ? 'WiLoR' : wl === 'starting' ? 'WiLoR starting…' : 'MediaPipe');
  const cams = (st && st.cameras) || [];
  $('#st-cams-wrap').classList.toggle('hidden', !cams.length);
  set('#st-cams', cams.map((c) => `${c.name.replace('camera', 'cam ')} ${c.aligned ? 'lined up' : c.seen ? '(press C)' : '(no picture)'}`).join(', '));
}

// ---------------------------------------------------------------- settings
const SETTINGS = [
  ['Face', [
    ['expressiveness', 'range', 'Expressiveness', 'Exaggerate or soften all face shapes', 0.5, 2, 0.05],
    ['auto_range', 'toggle', 'Auto-range', 'Stretch each shape to your own maximum'],
    ['link_blinks', 'toggle', 'Link blinks', 'Both eyes blink together unless you clearly wink'],
    ['face_refiner', 'toggle', 'Your face model', 'From “Train your face”, if you made one'],
    ['mouth_model', 'toggle', 'Tongue, cheek puff, funnel', 'Learned in “Train your face”: only the shapes it recognised reliably']]],
  ['Lip sync', [
    ['mic_lipsync', 'toggle', 'Microphone lip sync', 'Your voice helps the mouth'],
    ['mic_weight', 'range', 'Voice share of the vowel', '0 = camera only, 1 = voice only (while speaking)', 0, 1, 0.05],
    ['speech_detection', 'toggle', 'Speech detection', 'Only your voice opens the mouth, not typing, fans or music'],
    ['noise_suppression', 'toggle', 'Noise suppression', 'For loud rooms; the vowel guess gets a bit less exact']]],
  ['Motion', [
    ['smoothing', 'range', 'Smoothing', 'Left: smoother · right: snappier', 0.4, 2.5, 0.05],
    ['predict_body_ms', 'range', 'Body look-ahead (ms)', 'Torso, head and arms move this much earlier', 0, 40, 5],
    ['predict_ms', 'range', 'Look-ahead for everything (ms)', 'Also hands and fingers; makes them shakier', 0, 60, 5],
    ['body_follows_head', 'toggle', 'Body follows head', 'With your shoulders out of view, the upper body turns a little with the head'],
    ['no_clip', 'toggle', 'No clipping', 'Hands and arms stop at the body and head'],
    ['body_size', 'range', 'Body thickness', 'For no clipping: higher for bulky avatars or clothes', 0.7, 1.6, 0.05],
    ['idle', 'toggle', 'Idle motion', 'Breathing, blinks, easing back when tracking is lost'],
    ['root_motion', 'toggle', 'Move the avatar with you', 'Stepping closer or sideways moves it'],
    ['mirror_avatar', 'toggle', 'Mirror the avatar', 'Raise your right hand and the avatar raises its left']]],
  ['Camera and distance', [
    ['camera_fov', 'range', 'Camera field of view', 'How wide your webcam sees, side to side: sets how far away you are on the avatar', 40, 110, 1],
    ['eye_spacing_cm', 'range', 'Your eye spacing', 'Pupil to pupil (most adults 5.5–7 cm)', 5, 7.5, 0.1]]],
  ['Performance and outputs', [
    ['max_fps', 'range', 'Frame-rate cap', '0 = as fast as the camera; lower leaves CPU for games and OBS', 0, 30, 5],
    ['gpu_tracking', 'toggle', 'GPU tracking', 'Body and face models on the GPU: less delay (from the next start)'],
    ['wholebody', 'toggle', 'GPU whole-body model', 'Keeps left and right right when your hands cross'],
    ['wilor_fingers', 'toggle', 'WiLoR fingers', '3D hand model for finger shapes (GPU helper, if installed)'],
    ['vts', 'toggle', 'VTube Studio (Live2D)', 'Send face tracking to VTube Studio on port 8001']]],
  ['Camera picture', [
    ['show_mesh', 'toggle', 'Face mesh', 'The points tracked on your face'],
    ['show_body', 'toggle', 'Body skeleton', 'Body and hand points'],
    ['show_face_monitor', 'toggle', 'Face monitor', 'Mouth close-up and live face shapes'],
    ['show_panel', 'toggle', 'Depth panel', 'Distance and the side view']]],
];
const fmt = (k, v) => k === 'max_fps' ? (+v ? `${v}` : 'off') : k.startsWith('predict') ? `${v}`
  : k === 'camera_fov' ? `${Math.round(v)}°` : k === 'eye_spacing_cm' ? `${(+v).toFixed(1)} cm` : (+v).toFixed(2);
function buildSettings() {
  const grid = $('#settings-grid');
  let html = '';
  for (const [title, items] of SETTINGS) {
    html += `<div class="card"><h2>${esc(title)}</h2>`;
    for (const [key, type, label, help, min, max, step] of items) {
      const ctl = type === 'toggle'
        ? `<label class="switch"><input type="checkbox" data-set="${key}"><span></span></label>`
        : `<input type="range" data-set="${key}" min="${min}" max="${max}" step="${step}"><span class="val" data-val="${key}"></span>`;
      html += `<div class="row ${type}"><div class="label">${esc(label)}<small>${esc(help)}</small></div><div class="ctl">${ctl}</div></div>`;
    }
    if (title === 'Motion') html += `<div class="actions"><button class="btn" id="btn-reset-depth" title="When the avatar's distance looks wrong">Reset depth</button></div>`;
    if (title === 'Camera and distance') html += `<div class="row"><div class="label">Measure it<small>Sit facing the camera, measure from the camera to your eyes, enter it: the field of view is worked out</small></div>
      <div class="ctl"><input type="number" id="fov-distance" min="20" max="400" step="1" placeholder="cm" style="width:70px"><button class="btn" id="btn-measure-fov">Measure</button></div></div>`;
    html += '</div>';
  }
  html += `<div class="card"><h2>Devices and start</h2>
    <div class="row"><div class="label">Camera frame rate<small>60 halves the camera's share of the delay, if it can</small></div><div class="ctl"><select class="plain" id="sel-fps"></select></div></div>
    <div class="row"><div class="label">Second camera<small>Experimental: another angle (press C to line it up)</small></div><div class="ctl"><select class="plain" id="sel-camera2"></select></div></div>
    <div class="row"><div class="label">Third camera</div><div class="ctl"><select class="plain" id="sel-camera3"></select></div></div>
    <div class="row"><div class="label">Start tracking when the app opens</div><div class="ctl"><label class="switch"><input type="checkbox" data-pref="autostart"><span></span></label></div></div>
    <div class="row ${app.tray ? '' : 'hidden'}" id="row-tray"><div class="label">Minimize to the tray<small>An icon by the clock; its menu starts and stops tracking</small></div><div class="ctl"><label class="switch"><input type="checkbox" data-pref="tray"><span></span></label></div></div>
  </div>
  <div class="card"><h2>WiLoR finger helper</h2><p class="muted small" style="margin:4px 0 8px">A 3D hand model on your GPU for better finger shapes (fists, fingers hiding each other). Optional: about 4 GB, installed into its own folder.</p>
    <div class="row"><div class="label">Status<small id="wilor-status">–</small></div><div class="ctl"><button class="btn" id="btn-wilor-setup">Install…</button></div></div>
    <label class="small" style="display:flex;gap:8px;align-items:flex-start;margin:6px 0"><input type="checkbox" id="wilor-licence">
      <span>I accept the licences: WiLoR's weights are non-commercial only (CC BY-NC-ND 4.0); the MANO hand model is for non-commercial research, education and artistic projects; the hand detector (ultralytics) is AGPL-3.0. MANO comes from its makers: register at mano.is.tue.mpg.de, download <i>Models &amp; Code</i>, and pick <code>mano_v1_2/models/MANO_RIGHT.pkl</code> when asked.</span></label>
    <div class="progress hidden" data-progress="wilor-setup"><div class="bar"><span></span></div><span class="ptext"></span></div>
    <div class="log hidden" data-log="wilor-setup"></div>
  </div>
  <div class="card"><h2>OBS</h2><p class="muted small" style="margin:4px 0 8px">Your avatar in OBS without a VTuber app: add a <b>Browser</b> source with this address (while tracking runs; press Refresh in the source after starting tracking).</p>
    <div class="row"><div class="label">Avatar<small id="obs-avatar-name">none picked: a skeleton shows</small></div><div class="ctl"><button class="btn" id="btn-obs-avatar">Choose .vrm…</button></div></div>
    <div class="row"><div class="label">Background</div><div class="ctl"><select class="plain" id="sel-obs-bg"><option value="">Transparent</option><option value="00ff00">Green (colour key)</option><option value="0000ff">Blue (colour key)</option></select></div></div>
    <div class="row"><div class="label">Framing</div><div class="ctl"><select class="plain" id="sel-obs-view"><option value="upper">Upper body</option><option value="full">Whole body</option><option value="face">Face</option></select></div></div>
    <div class="row"><input type="text" id="obs-url" readonly style="flex:1;font-family:ui-monospace,Consolas,monospace"><button class="btn" id="btn-obs-copy">Copy</button></div>
  </div>
  <div class="card"><h2>Updates</h2>
    <div class="row"><div class="label">Version<small id="version-detail"></small></div><div class="ctl"><span class="val" id="version-now"></span></div></div>
    <div class="row"><div class="label">Check for updates when the app opens<small>New versions come from the app's GitHub releases</small></div><div class="ctl"><label class="switch"><input type="checkbox" data-pref="check_updates"><span></span></label></div></div>
    <div class="row"><div class="label" id="update-state"></div><div class="ctl"><button class="btn" id="btn-check-updates">Check now</button></div></div>
  </div>
  <div class="card"><h2>Profiles</h2><p class="muted small" style="margin:4px 0 8px">A profile keeps these settings and your calibration, for example one per room or chair.</p>
    <div class="row"><input type="text" id="profile-name" placeholder="Desk, evening" style="flex:1"><button class="btn" id="btn-profile-save">Save</button></div>
    <div class="row"><select class="plain" id="profile-list" style="flex:1"></select><button class="btn" id="btn-profile-load">Load</button></div>
    <div class="actions"><button class="btn danger" id="btn-reset">Reset all settings</button></div>
  </div>
  <div class="card"><h2>Report a problem</h2><p class="muted small" style="margin:4px 0 8px">Puts together a bug report (what happened, version, system, settings, recent log lines, with personal details taken out). You read it first; nothing is sent until you submit it on GitHub yourself.</p>
    <textarea id="report-desc" rows="3" placeholder="What happened? What did you expect?" style="width:100%"></textarea>
    <div class="actions"><button class="btn" id="btn-report">Prepare report…</button></div>
  </div>
  <div class="card"><h2>Uninstall</h2><p class="muted small" style="margin:4px 0 8px">Removes Avatar Studio from this PC: the app, its unpacked files and caches, and its shortcuts. Your data (calibration, recordings, training) is kept unless you choose otherwise.</p>
    <div class="actions"><button class="btn danger" id="btn-uninstall">Uninstall Avatar Studio…</button></div>
  </div>`;
  grid.innerHTML = html;
  $$('[data-set]', grid).forEach((el) => {
    const key = el.dataset.set;
    el.addEventListener(el.type === 'range' ? 'input' : 'change', () => {
      const v = el.type === 'checkbox' ? el.checked : parseFloat(el.value);
      if (el.type === 'range') $(`[data-val="${key}"]`).textContent = fmt(key, v);
      changeSetting(key, v);
    });
  });
  $$('[data-pref]', grid).forEach((el) => (el.onchange = () => setPref({ [el.dataset.pref]: el.checked })));
  $('#sel-fps').onchange = (e) => setPref({ fps: e.target.value });
  $('#sel-camera2').onchange = (e) => setPref({ camera2: e.target.value });
  $('#sel-camera3').onchange = (e) => setPref({ camera3: e.target.value });
  $('#btn-reset-depth').onclick = () => { if (!send({ type: 'action', name: 'reset_depth' })) toast('Start tracking first'); };
  $('#btn-measure-fov').onclick = () => {
    const cm = parseFloat($('#fov-distance').value);
    if (!(cm >= 20 && cm <= 400)) { toast('Enter your distance from the camera in cm (20–400)'); return; }
    if (!send({ type: 'action', name: 'measure_fov', arg: cm })) toast('Start tracking first');
  };
  $('#btn-check-updates').onclick = async () => { app.update = await call('check_updates'); renderUpdate(); };
  $('#btn-profile-save').onclick = async () => {
    const name = $('#profile-name').value.trim();
    if (!name) { toast('Enter a name for the profile'); return; }
    if (!send({ type: 'save_profile', name })) await call('save_profile', name);
    toast(`Profile “${name}” saved`, 'ok'); $('#profile-name').value = '';
    setTimeout(loadSettings, 500);
  };
  $('#btn-profile-load').onclick = async () => {
    const name = $('#profile-list').value;
    if (!name) return;
    if (!send({ type: 'load_profile', name })) await call('load_profile', name);
    toast(`Profile “${name}” loaded`, 'ok'); setTimeout(loadSettings, 500);
  };
  $('#btn-reset').onclick = async () => {
    if (!confirm('Set every setting back to its default? Your calibration is kept.')) return;
    if (!send({ type: 'action', name: 'reset_settings' })) await call('reset_settings');
    setTimeout(loadSettings, 500);
  };
  $('#btn-uninstall').onclick = openUninstall;
  $('#sel-obs-bg').onchange = async (e) => { await setPref({ obs_bg: e.target.value }); fillObs(); };
  $('#sel-obs-view').onchange = async (e) => { await setPref({ obs_view: e.target.value }); fillObs(); };
  $('#btn-obs-copy').onclick = () => { $('#obs-url').select(); document.execCommand('copy'); toast('OBS address copied', 'ok'); };
  $('#btn-obs-avatar').onclick = async () => {
    const r = await call('choose_obs_avatar');
    if (r.ok) { app.prefs.obs_avatar = r.name; fillObs(); toast(`OBS avatar: ${r.name} (press Refresh in the OBS source)`, 'ok'); }
    else if (!r.cancelled) toast(r.error, 'bad');
  };
  fillObs();
  $('#btn-wilor-setup').onclick = async () => {
    if (!$('#wilor-licence').checked) { toast('Tick the licence box first'); return; }
    const r = await call('run_tool', 'wilor-setup');
    if (!r.ok && !r.cancelled) toast(r.error, 'bad');
  };
  fillWilor();
  $('#btn-report').onclick = () => openReport($('#report-desc').value);
}

async function fillWilor() {  // Settings > WiLoR finger helper
  if (!$('#wilor-status')) return;
  const s = await call('wilor_state'), job = app.jobs && app.jobs['wilor-setup'];
  const busy = !!(job && job.running);
  $('#wilor-status').textContent = busy ? 'Installing…' : s.usable ? (s.installed ? `Installed in ${s.folder}` : 'Installed (your own .venv_wilor)')
    : s.windows ? 'Not installed: fingers come from MediaPipe' : 'Not installed (the automatic setup is for Windows; see the README)';
  $('#btn-wilor-setup').textContent = s.usable ? 'Repair…' : 'Install…';
  $('#btn-wilor-setup').disabled = busy || !s.windows;
}

function fillObs() {  // Settings > OBS: the browser-source address for the chosen look
  const p = app.prefs || {}, bg = p.obs_bg || '', view = p.obs_view || 'upper';
  if (!$('#obs-url')) return;
  $('#sel-obs-bg').value = bg; $('#sel-obs-view').value = view;
  $('#obs-url').value = `http://127.0.0.1:8000/?obs${view !== 'upper' ? `&view=${view}` : ''}${bg ? `&bg=${bg}` : ''}`;
  $('#obs-avatar-name').textContent = p.obs_avatar ? p.obs_avatar : 'none picked: a skeleton shows';
}

// ---------------------------------------------------------------- report a problem (core/bugreport.py)
async function openReport(description = '') {
  const r = await call('bug_report', description);
  app.report = r;
  $('#report-text').value = r.text;
  $('#report-note').textContent = !r.url ? 'This build has no GitHub repository set: copy the report and send it however you like.'
    : r.short ? 'The full report is too long for a link: GitHub opens with fewer log lines. Copy the full text and paste it there if you like.'
      : 'GitHub opens in your browser with this filled in. Check it, then press Submit there (a free GitHub account is needed).';
  $('#btn-report-open').disabled = !r.url;
  $('#report-dialog').classList.remove('hidden');
}
$('#btn-report-open').onclick = async () => { const r = await call('open_url', app.report.url); if (!r.ok) toast(r.error, 'bad'); };
$('#btn-report-copy').onclick = () => { $('#report-text').select(); document.execCommand('copy'); toast('Report copied', 'ok'); };
$('#btn-report-close').onclick = () => $('#report-dialog').classList.add('hidden');

// ---------------------------------------------------------------- first-run guide (once; Getting started reopens it)
const GUIDE = [
  ['Welcome to Avatar Studio', () => `
    <p>Your webcam moves a VTuber avatar: face, expressions, body and every finger. Everything runs on this PC; nothing
    is uploaded.</p><p class="muted">Four short steps get you moving. You can skip and come back from <b>Getting started</b>.</p>`],
  ['Your camera', () => `
    <div class="row"><div class="label">Camera</div><div class="ctl"><select class="plain" id="guide-camera"></select></div></div>
    <ul class="muted small guide-tips">
      <li>Light your face from the front (a window or lamp behind the screen, not behind you).</li>
      <li>Camera at about eye level, an arm's length away.</li>
      <li>Head, shoulders and both hands in the picture, for finger tracking.</li>
    </ul>`],
  ['Start tracking and set your neutral pose', () => `
    <p>Start tracking, then sit as you usually do, relax your face, look at the screen and press <b>Neutral pose</b>.
    Hold still for a second.</p>
    <div class="actions"><button class="btn primary" id="guide-start"><svg class="i"><use href="#i-play"/></svg>Start tracking</button>
      <button class="btn" id="guide-neutral"><svg class="i"><use href="#i-target"/></svg>Neutral pose (C)</button></div>
    <p class="muted small" id="guide-track-state"></p>`],
  ['Connect your VTuber app', () => `
    <p>In VSeeFace, VNyan, Warudo or VMagicMirror turn on <b>VMC receiving on port 39539</b>. For a Live2D model, open
    VTube Studio and allow the plugin when it asks.</p>
    <div class="conn" id="guide-conn"></div>
    <div class="actions"><button class="btn" id="guide-conn-check"><svg class="i"><use href="#i-cast"/></svg>Check again</button></div>
    <p class="muted small">No VTuber app? The built-in 3D viewer shows your avatar (drop your .vrm on it); in OBS add a browser source with the address from Settings › OBS.</p>`],
  ['You\'re set', () => `
    <p>Your avatar follows you now. For an even closer match, <b>Getting started</b> has a few more one-time steps:
    your vowels and voice for lip sync, open hands and fists, eye gaze, and a 2-minute face training.</p>
    <p class="muted small">Something not working? Settings › <b>Report a problem</b> prepares a bug report for you to check and send.</p>`],
];
let guideStep = 0;
async function maybeShowGuide() {
  const r = await call('checklist');
  if (r.steps[1] && r.steps[1].done) { setPref({ guide_done: true }); return; }  // (already calibrated: not new)
  openGuide();
}
function openGuide() { guideStep = 0; renderGuide(); $('#guide').classList.remove('hidden'); }
function closeGuide(page) {
  $('#guide').classList.add('hidden');
  setPref({ guide_done: true });
  if (page) showPage(page);
}
function renderGuide() {
  const [title, body] = GUIDE[guideStep];
  $('#guide-title').textContent = title;
  $('#guide-body').innerHTML = body();
  $('#guide-dots').innerHTML = GUIDE.map((_, i) => `<span class="${i === guideStep ? 'on' : i < guideStep ? 'done' : ''}"></span>`).join('');
  $('#btn-guide-back').classList.toggle('hidden', guideStep === 0);
  $('#btn-guide-next').textContent = guideStep === GUIDE.length - 1 ? 'Finish' : guideStep === 0 ? 'Let\'s go' : 'Next';
  $('#btn-guide-skip').classList.toggle('hidden', guideStep === GUIDE.length - 1);
  if ($('#guide-camera')) {
    const src = $('#sel-camera');
    $('#guide-camera').innerHTML = src.innerHTML; $('#guide-camera').value = src.value;
    $('#guide-camera').onchange = (e) => { src.value = e.target.value; setPref({ camera: e.target.value }); };
  }
  if ($('#guide-start')) {
    const state = () => {
      const on = engineRunning();
      $('#guide-start').disabled = on; $('#guide-neutral').disabled = !on;
      $('#guide-track-state').textContent = on ? 'Tracking is on: the Live page shows your camera.' : '';
    };
    $('#guide-start').onclick = async () => { await start('live'); $('#guide').classList.remove('hidden'); state(); };
    $('#guide-neutral').onclick = () => { sendKey('c'); toast('Hold still: setting your neutral pose', 'ok'); };
    state(); app.guideTick = setInterval(() => ($('#guide-start') ? state() : clearInterval(app.guideTick)), 700);
  }
  if ($('#guide-conn')) {
    checkConnections('#guide-conn');
    $('#guide-conn-check').onclick = () => checkConnections('#guide-conn');
  }
}
$('#btn-guide-next').onclick = () => {
  if (guideStep === GUIDE.length - 1) { closeGuide('start'); return; }
  guideStep++; renderGuide();
};
$('#btn-guide-back').onclick = () => { guideStep = Math.max(0, guideStep - 1); renderGuide(); };
$('#btn-guide-skip').onclick = () => closeGuide();
$('#btn-guide').onclick = openGuide;

// ---------------------------------------------------------------- is a VTuber app listening? (engine/connections.py)
const CONN_HELP = {
  vmc: 'Nothing is listening on port 39539. In your VTuber app turn on VMC / OSC receiving on port 39539 (VSeeFace: General settings › OSC/VMC receiver › Enable; VNyan and Warudo: VMC receiver).',
  vts: 'VTube Studio isn\'t running (or its plugin API is off: Settings › Start API, port 8001). Only needed for Live2D models.',
};
async function checkConnections(target = '#conn-status') {
  const el = $(target);
  if (!el) return null;
  el.innerHTML = '<span class="muted small">Checking…</span>';
  const r = await call('check_connections');
  if (r.error) { el.innerHTML = `<p class="error small">${esc(r.error)}</p>`; return r; }
  const row = (key, label) => {
    const c = r[key];
    return c.listening
      ? `<div class="conn-row ok"><svg class="i"><use href="#i-checks"/></svg><span><b>${esc(c.app || 'An app')}</b> is ready for ${label} on port ${c.port}</span></div>`
      : `<div class="conn-row"><svg class="i"><use href="#i-info"/></svg><span>${esc(CONN_HELP[key])}</span></div>`;
  };
  el.innerHTML = row('vmc', 'the avatar (VMC)') + row('vts', 'face tracking (VTube Studio)');
  return r;
}
$('#btn-conn-check').onclick = () => checkConnections();

// ---------------------------------------------------------------- uninstall (core/uninstall.py)
async function openUninstall() {
  const p = await call('uninstall_plan');
  const total = p.app.reduce((s, [, mb]) => s + mb, 0);
  $('#uninstall-list').innerHTML = p.app.map(([path, mb]) => `<li><code>${esc(path)}</code> <span class="muted">${mb} MB</span></li>`).join('')
    || '<li class="muted">Nothing found</li>';
  $('#uninstall-sub').textContent = p.packaged
    ? `Frees about ${total} MB. Avatar Studio closes, and its files are gone a few seconds later.`
    : 'This is a development run: there is nothing installed to remove.';
  $('#uninstall-data-path').textContent = p.data ? `${p.data[0]} (${p.data[1]} MB)` : '';
  $('#uninstall-data').checked = false;
  $('#uninstall-error').classList.add('hidden');
  $('#btn-uninstall-now').disabled = !p.packaged;
  $('#uninstall-dialog').classList.remove('hidden');
}
$('#btn-uninstall-cancel').onclick = () => $('#uninstall-dialog').classList.add('hidden');
$('#btn-uninstall-now').onclick = async () => {
  $('#btn-uninstall-now').disabled = true;
  const r = await call('uninstall', $('#uninstall-data').checked);
  if (r.ok) { $('#uninstall-sub').textContent = 'Uninstalling… Avatar Studio is closing.'; return; }
  $('#uninstall-error').textContent = r.error; $('#uninstall-error').classList.remove('hidden');
  $('#btn-uninstall-now').disabled = false;
};
async function loadSettings() {
  if (!tracking()) {
    const r = await call('get_settings');
    app.settings = r.settings; app.profiles = r.profiles; app.hotkeys = r.hotkeys;
  }
  fillSettings();
  fillWilor();
}
function fillSettings() {
  const s = app.settings || {};
  $$('[data-set]').forEach((el) => {
    const v = s[el.dataset.set];
    if (v === undefined) return;
    if (el.type === 'checkbox') el.checked = !!v;
    else { el.value = v; $(`[data-val="${el.dataset.set}"]`).textContent = fmt(el.dataset.set, v); }
  });
  const ON = ['tray', 'check_updates'];  // (on unless switched off)
  $$('[data-pref]').forEach((el) => {
    const k = el.dataset.pref;
    el.checked = app.prefs[k] === undefined ? ON.includes(k) : !!app.prefs[k];
  });
  $('#version-now').textContent = app.version || '';
  $('#version-detail').textContent = app.packaged ? 'Packaged app' : 'Development run';
  renderUpdate();
  $('#row-tray').classList.toggle('hidden', !app.tray);
  const list = $('#profile-list');
  options(list, app.profiles.length ? app.profiles.map((p) => [p, p]) : [['', 'No profiles yet']], list.value);
  $('#settings-lead').textContent = tracking() ? 'Changes apply at once and are kept.'
    : 'Tracking is off: changes are saved and used when it starts.';
  fillDevices();
  fillObs();
}
function changeSetting(key, value) {
  app.editing = performance.now();
  app.settings = { ...(app.settings || {}), [key]: value };
  if (!(tracking() && send({ type: 'set', values: { [key]: value } }))) call('set_settings', { [key]: value });
  render();
}

// ---------------------------------------------------------------- getting started
const STEP_TEXT = [
  ['Start tracking', 'Then turn on VMC receiving (port 39539) in your VTuber app.', ['Start tracking', () => start('live')]],
  ['Neutral pose', 'Sit as usual, look at the screen and press Neutral pose (C).', ['Calibrate', () => { showPage('live'); sendKey('c'); }]],
  ['Vowels and voice', 'Say A, I, U, E, O when asked. It also learns your voice for lip sync.', ['Vowels', () => { showPage('live'); sendKey('v'); }]],
  ['Hands: open, then fists', 'Hold both hands open, then make fists, 3 seconds each.', ['Hands', () => { showPage('live'); sendKey('k'); }]],
  ['Eye gaze (optional)', 'Follow the dots with your eyes, head still.', ['Eye gaze', () => { showPage('live'); sendKey('g'); }]],
  ['Your face (optional, about 2 min)', 'Makes smiles, winks and brows read like yours.', ['Personalize', () => showPage('personalize')]],
];
async function loadChecklist() {
  const r = await call('checklist');
  const next = r.steps.findIndex((s) => !s.done);
  $('#steps').innerHTML = r.steps.map((s, i) => {
    const [title, text] = STEP_TEXT[i] || [s.what, s.how];
    const cls = s.done ? 'done' : i === next ? 'next' : '';
    return `<li class="step ${cls}"><div class="mark">${s.done ? '✓' : i + 1}</div><div class="body"><h3>${esc(title)}</h3>
      <p>${esc(text)}</p></div>${STEP_TEXT[i] && !s.done ? `<button class="btn ${i === next ? 'primary' : ''}" data-step="${i}">${esc(STEP_TEXT[i][2][0])}</button>` : ''}</li>`;
  }).join('');
  $$('[data-step]').forEach((b) => (b.onclick = () => STEP_TEXT[+b.dataset.step][2][1]()));
  $('#info').innerHTML = r.info.map((x) => `<div class="row"><div class="label">${esc(x.text)}</div><span class="dot ${x.ok ? 'ok' : ''}"></span></div>`).join('');
}

// ---------------------------------------------------------------- updates (from the app's GitHub releases)
function renderUpdate() {
  const u = app.update || { state: 'idle' }, info = u.info || {};
  const busy = u.state === 'downloading' || u.state === 'installing';
  const btn = $('#btn-update');
  btn.classList.toggle('hidden', !(u.state === 'available' || busy));
  $('span', btn).textContent = u.state === 'downloading' ? `Updating… ${Math.round((u.progress || 0) * 100)}%`
    : u.state === 'installing' ? 'Restarting…' : `Update to ${info.version}`;
  const dlg = $('#update-dialog');
  if (!dlg.classList.contains('hidden')) {
    $('#update-title').textContent = `Avatar Studio ${info.version || ''} is available`;
    $('#update-sub').textContent = `You have version ${app.version}. Updating downloads about ${Math.round((info.patch_size || info.size || 0) / 2 ** 20)} MB, `
      + 'stops tracking and restarts the app; the new version prepares itself once on its first start.';
    $('#update-notes').textContent = info.notes || '';
    const p = $('#update-progress');
    p.classList.toggle('hidden', !busy);
    $('.bar span', p).style.width = `${Math.round((u.state === 'installing' ? 1 : u.progress || 0) * 100)}%`;
    $('.ptext', p).textContent = u.state === 'installing' ? 'Restarting with the new version…'
      : `Downloading ${u.done_mb || 0} of ${u.total_mb || Math.round((info.patch_size || info.size || 0) / 2 ** 20)} MB`;
    $('#update-error').textContent = u.error || '';
    $('#update-error').classList.toggle('hidden', !u.error);
    $('#btn-update-now').disabled = busy;
  }
  const st = $('#update-state');
  if (st) {
    st.textContent = !app.updatesFrom ? 'Updates are off in this build (no GitHub repository in avatarstudio/version.py)'
      : u.state === 'checking' ? 'Checking…' : u.state === 'latest' ? 'You have the latest version'
        : u.state === 'available' ? `Version ${info.version} is available` : busy ? 'Updating…'
          : u.state === 'error' ? u.error : app.packaged ? '' : 'Development run: updates install into the packaged app';
  }
}
$('#btn-update').onclick = () => { $('#update-dialog').classList.remove('hidden'); renderUpdate(); };
$('#btn-update-later').onclick = () => $('#update-dialog').classList.add('hidden');
$('#btn-update-now').onclick = async () => {
  const r = await call('install_update');
  if (!r.ok) { app.update = { ...app.update, error: r.error }; renderUpdate(); }
};

// ---------------------------------------------------------------- custom gestures
async function loadGestures() {
  const g = await call('gestures_state');
  const opts = (sel) => ['<option value="">No expression</option>', ...g.expressions.map((x) =>
    `<option${x === sel ? ' selected' : ''}>${esc(x)}</option>`)].join('');
  $('#gesture-list').innerHTML = g.gestures.map((x) => `<div class="gesture">
      <div class="gname">${esc(x.name)}<small>${x.samples} samples${x.name === 'none' ? ' · relaxed hands' : ''}</small></div>
      ${x.name === 'none' ? '' : `<select class="plain" data-gaction="${esc(x.name)}" title="Shown while you hold this sign">${opts(g.actions[x.name])}</select>`}
      <button class="btn" data-grecord="${esc(x.name)}" title="Record 8 more seconds">More</button>
      <button class="btn ghost" data-gdelete="${esc(x.name)}" title="Delete this gesture's recordings">Delete</button></div>`).join('');
  const names = g.gestures.map((x) => x.name);
  $('#gesture-state').textContent = !names.length ? 'Your own hand signs show expressions'
    : !names.includes('none') ? 'Record “none” too, then train'
      : names.length < 2 ? 'Record at least one sign, then train'
        : g.up_to_date ? 'Trained: the studio uses them' : g.trained ? 'Changed since training: train again' : 'Not trained yet';
  $$('[data-gaction]').forEach((s) => (s.onchange = () => call('set_gesture_action', s.dataset.gaction, s.value)));
  $$('[data-grecord]').forEach((b) => (b.onclick = () => start('gesture', b.dataset.grecord)));
  $$('[data-gdelete]').forEach((b) => (b.onclick = async () => {
    if (!confirm(`Delete the recordings of “${b.dataset.gdelete}”?`)) return;
    await call('delete_gesture', b.dataset.gdelete); loadGestures();
  }));
}
$('#btn-gesture-add').onclick = () => {
  const name = $('#gesture-name').value.trim().replace(/\s+/g, '_');
  if (!name) { toast('Enter a name for the gesture, for example peace'); return; }
  $('#gesture-name').value = '';
  start('gesture', name);
};

// ---------------------------------------------------------------- go
buildSettings();
connect();
poll();
apiReady.then(() => call('get_settings')).then((r) => {
  if (!app.status) { app.settings = r.settings; app.profiles = r.profiles; app.hotkeys = r.hotkeys; }
  render();
});
setInterval(() => { if (app.page === 'start') loadChecklist(); }, 4000);
