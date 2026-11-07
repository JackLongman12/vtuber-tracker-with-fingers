# Webcam Avatar Studio

Webcam-only motion capture for VTuber avatars and general 3D work: face, eyes, expressions, head, full body,
arms, hands, every finger and depth. It drives VSeeFace, VNyan, Warudo, VMagicMirror or anything else that speaks the
VMC protocol, streams to a live 3D viewer, and exports BVH motion for Blender. Recordings can also be processed
offline (`avatarstudio/mocap.py`) into clean motion, 3D body points, joint angles and face curves for animation and movement
analysis.

Developers: **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** explains the project layout, setup, building and
publishing updates.

A free hobby project for personal, non-commercial use: the code is MIT, but some of the models and the data they
were trained on are non-commercial (see [License](#license)).

## Download
**[Download AvatarStudio.exe](https://github.com/JackLongman12/vtuber-tracker-with-fingers/releases/latest/download/AvatarStudio.exe)**
(Windows 10/11, about 500 MB): one file with everything inside - models, 3D viewer, all tools. No Python, no setup:
download it and double-click. The first start unpacks it once (a progress window, a few seconds); later starts are quick, and it
offers updates by itself. Windows may warn about an unknown publisher (the app isn't code-signed): *More info* →
*Run anyway*. Optional extra: the WiLoR finger helper needs its own setup (see [WiLoR fingers](#wilor-fingers-3d-hand-model-gpu)).

## New in v1.1
(Trial versions can be published as **pre-releases** from a branch, `tools/release.py --prerelease --branch ...`:
the app never updates to one by itself, and the Download link above stays on the newest full release.)
- **Setup guide** on first start: camera, lighting tips, tracking and neutral pose, connecting your VTuber app.
  Getting started → *Show the setup guide* opens it again.
- **Connection check** (Check setup page, and in the guide): shows which app is listening for the avatar on port
  39539 (VSeeFace, VNyan, Warudo, ...) and whether VTube Studio's plugin API is on, or how to turn them on.
- **Report a problem** (Settings): puts together a bug report (version, system, settings, recent log lines, with
  your user name, PC name, folder paths, email and network addresses taken out). You read all of it first; GitHub
  opens with it filled in, and nothing is sent until you press Submit there.
- **Avatar in OBS without a VTuber app** (Settings → OBS): a browser-source address showing only your avatar, on a
  transparent background (or green / blue for a colour key), framed on the upper body, the whole body or the face.
  Pick your `.vrm` there once; the studio serves it to OBS (a browser source can't have a file dropped on it).
- **Camera field of view and eye spacing** (Settings → Camera and distance): the absolute distance used to be
  10–15% off unless `--fov` and `--ipd` were set on the command line. Now they're settings, and *Measure it* works
  the field of view out from your real distance to the camera. Changing them while tracking rescales the distance
  (and the calibration's) so the avatar doesn't jump.
- **Crash notice:** when tracking, a tool or the WiLoR finger helper stops by itself (or a part of tracking fails and
  it carries on), the app says what stopped, with its last message and a **Report this problem** button that opens
  the bug report already filled in with it.
- **Phantom hands dropped:** turned away from the camera, the hand model could "find" a hand in your hair. The avatar
  froze its head as if a hand covered your face and reached its arm up there. A hand further than 3.5 body sizes
  from both of the body model's wrists is now ignored (real hands on the test takes: at most 2.7).

## Uninstalling
In the app: **Settings → Uninstall Avatar Studio…** (classic launcher: *Check → Uninstall...*; or
`AvatarStudio.exe uninstall` in a console). Avatar Studio closes, and a few seconds later everything it put on your PC
is gone. Your data (calibration, recordings, training) is kept unless you tick *Also delete my data*; then it goes to
the Recycle Bin, so you can still restore it.

What it removes, if you'd rather do it by hand (it has no installer and changes nothing in the registry):

| Where | What |
|---|---|
| your `AvatarStudio.exe` | the app |
| `%LOCALAPPDATA%\AvatarStudio` | the unpacked app (about 800 MB per version) and window caches |
| `%APPDATA%\AvatarStudio` | your data: calibration, settings, recordings, face / gesture training, profiles |
| Start menu and desktop | `Avatar Studio.lnk`, if you made the shortcuts |
| `%TEMP%\AvatarStudio-wilor` | the WiLoR helper's scripts, if you used WiLoR (its own `.venv_wilor` you set up yourself) |

## Quick start
Double-click **`AvatarStudio.exe`** (or, after building it yourself, `Distributions\AvatarStudio.exe`) - the whole app is that one file: one window with your camera,
your 3D avatar, the settings and every tool (calibration, avatar test, latency test, face training, gestures,
smoothing, motion capture). From the project folder:
```bash
.venv\Scripts\python.exe app.py
```
Or only the tracking engine, with its own camera window (`app.py studio --help` lists its options):
```bash
.venv\Scripts\python.exe app.py studio
```

1. Sit or stand normally, relax your face, look at the screen and press **C** to calibrate (hold still for about 1 s).
2. Press **V** for the vowel calibration: say each vowel (A, I, U, E, O) out loud when prompted and hold it
   for 1.5 s. This records your mouth shapes *and* your voice, and is saved in `calibration.json`.
3. In your VTuber app, turn on **VMC protocol receiving** on port **39539** (VSeeFace: General settings →
   OSC/VMC receiver). For a Live2D model in **VTube Studio**, see below instead.
4. **Settings** and **3D viewer**: the launcher's buttons open them in their own windows (in a browser or OBS:
   http://127.0.0.1:8000/settings.html · http://127.0.0.1:8000). Click *Load VRM…* or drop your `.vrm` file onto the
   viewer to see your own avatar.

Keys (studio window): `c` calibrate · `v` vowels · `g` eye gaze · `1`–`9` expression hotkeys · `f` face monitor · `q` quit ·
`m` face mesh · `b` body skeleton · `h` side panel. **Ctrl+Alt+1–9** toggle expressions from any window.

## Your data survives new builds
Packaged builds keep everything you make (calibration, settings, face / mouth training, smoothing, profiles,
gestures, takes) in **`%APPDATA%\AvatarStudio`** (`avatarstudio/core/paths.py`), shared by every build. Development runs keep it
in the project's `data/` folder (not in git); `AVATARSTUDIO_DATA` overrides both. Settings and calibration are written atomically (a crash
mid-save can't leave a half-written file), and a damaged file is set aside as `<name>.broken` with defaults used
instead of stopping the app. Settings files are versioned: changed defaults are applied once to older files.

## Updates
The app checks for a new version when it opens (Settings > Updates switches that off and has *Check now*). When
there is one, **Update to x.y** appears in the top bar: its dialog shows what's new, *Update and restart*
downloads it (checked against its SHA-256 checksum), stops tracking, swaps the app file and starts the new
version, which prepares itself once like any new build. Your data isn't touched. Updates come from the GitHub
releases of the repository in `avatarstudio/version.py`; publishing one is `tools/release.py`
(docs/DEVELOPMENT.md).

## One-file app
`tools/build_dist.py` makes **one file: `Distributions/AvatarStudio.exe`** - nothing else next to it (no
Python needed, no .bat files, no folders). The app (runtime, models, 3D viewer) is zipped onto the end of a small
launcher (`packaging/single_exe.cs`, compiled with the C# compiler that comes with Windows). The first start of a
version unpacks it once (progress window) into `%LOCALAPPDATA%\AvatarStudio\app\<version>`; every later
start goes straight there. Unpacking runs on up to 6 cores at once (each thread with its own reader of the zip), and the
progress window stays until the app's own window is up. On this laptop, a first start from a fresh copy: unpacked in
3.0–3.5 s instead of 7.1–7.6 s, app window at 4.5–4.9 s instead of 8.6–9.0 s, and no moment with no window on screen
(it was 2 s before). Before the progress window shows, Windows checks a newly downloaded 500 MB file once (2–5 s here);
the launcher can't start any earlier than that. Later starts: 1.3 s to the app window. (PyInstaller's own one-file mode unpacks ~800 MB to a temporary folder on *every*
start, which made each start slow.) A new build replaces the previous one in `Distributions`; the unpacked copies keep
the current and the previous version.
Arguments work as before: `AvatarStudio.exe studio --test-pattern`. The name stays the same from build to build, so the launcher's *Start menu + desktop shortcut* keeps working; the version is in its title bar. Old folder builds were removed;
anything you had made inside them is in `%APPDATA%\AvatarStudio\old_builds_backup`. About 500 MB, mostly the
GPU whole-body model and MediaPipe/ONNX Runtime; PyTorch isn't included (the WiLoR finger helper uses the
project's `.venv_wilor`, found next to the `Distributions` folder).

## macOS and Linux
The studio, the launcher, the viewer / settings windows and motion capture also run from source on macOS and
Linux (the one-file app is Windows-only so far). **Not tried on a Mac or a Linux PC yet:** the code paths are
written, Windows was checked to behave exactly as before, and the macOS / Linux way of the viewer windows was tried
on Windows. Please report what happens.
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```
Linux also needs `sudo apt install libportaudio2` (microphone) and `.venv/bin/pip install "pywebview[gtk]"`
(viewer windows; otherwise they open in the browser). For the GPU on Linux, replace `onnxruntime` with
`onnxruntime-gpu` (NVIDIA, CUDA 12), `onnxruntime-rocm` (AMD) or `onnxruntime-openvino` (Intel).

| | Windows | macOS | Linux |
|---|---|---|---|
| GPU models (GPU tracking, whole-body model; `avatarstudio/core/os_support.py`) | DirectML | Core ML: Metal GPU or Neural Engine | CUDA, ROCm / MIGraphX, OpenVINO |
| WiLoR fingers (PyTorch, `.venv_wilor`) | CUDA | Metal (`mps`) | CUDA (NVIDIA; AMD's ROCm build too) |
| Camera | DirectShow | AVFoundation | V4L2 |
| Microphones listed | WASAPI | Core Audio | the default API (ALSA / PulseAudio) |
| Viewer / settings windows | Edge WebView2 | WKWebView | WebKitGTK or Qt WebEngine |
| Background slow-down switched off | power throttling | App Nap | (none) |
| Your data (packaged) | `%APPDATA%\AvatarStudio` | `~/Library/Application Support/AvatarStudio` | `~/.local/share/AvatarStudio` |

- **GPU on a Mac:** switch on *GPU tracking* (settings page, from the next start) to run the body and face models
  through Core ML. Core ML picks the Metal GPU or the Neural Engine per model; `AVATARSTUDIO_COREML_UNITS=CPUAndGPU`
  keeps it on Metal and `CPUAndNeuralEngine` on the Neural Engine, to compare. Which is fastest is still to be measured.
- **macOS asks once** for the camera and the microphone, and the Ctrl+Alt hotkeys need *Privacy & Security >
  Accessibility*. The OS requires this; it can't be avoided.
- Tools started from the launcher get no console window: their messages are in `logs/<tool>.log` in your data
  folder. No tray icon (the launcher minimizes like any window) and no Start menu shortcut button.
- Noise suppression needs RNNoise built for that OS (`models/librnnoise.dylib` / `models/librnnoise.so`, from the
  pyrnnoise wheel; not included yet). Without it the voice is used as it is.
- Linux: the Ctrl+Alt hotkeys work on X11, not Wayland. The WiLoR environment is found at `.venv_wilor/bin/python`.

## Speed and robustness (latest round)
Replaying `take.mp4` through the whole studio (headless; frame → avatar data sent):
| | median | 95th | worst frames |
|---|---|---|---|
| before (body model "full") | 34 ms | 121 ms | every fingertip pinch stalled the avatar |
| before (body model "heavy") | 65 ms | 76 ms | |
| after the first round | 27 ms | 36 ms | |
| **now** (one camera, WiLoR on) | **25 ms** | **34 ms** | |
| two cameras (experimental), WiLoR on | 31 ms | 41 ms | |
- **Fingertip pinch solver** (`rig.solve_pinch`): a general-purpose search took ~150 ms per pinching hand (430 ms
  with profiling), on 15% of the take's frames. Now Levenberg-Marquardt on a lean tip function: ~1 ms, same or
  better result (gap 0.0 cm, 90th 0.0 cm). Rig per frame: mean 26 → 2.5 ms, worst 423 → 26 ms.
- **Rig maths**: numpy's general cross product, clip and norm on single 3-vectors were most of the remaining rig
  time; plain formulas (checked equal to 1e-15) and forward kinematics of only the bones a solver needs.
- **Body model "full"** again (heavy: twice the tracking time for a moderately better torso turn).
- **GPU whole-body model only when left/right is in doubt** (hands close together, or the body model can't vouch
  for a hand) and 0.5 s after. Running all the time, it shared the GPU with WiLoR and made fingers ~20 ms later.
  Same left/right decisions as running it on every frame (1692/1692 hands, 0 flips); without it at all, 6 swaps.
- **Preview window on its own thread**: drawing it took ~4 ms + 1-2 ms to show, on the tracking loop.
- **3D viewer glide**: smooths over half the time between poses (16 ms) instead of a fixed 30 ms.
- **Robustness**: one failing part on one frame (tracking, WiLoR, gestures, mouth model, rig, an output) is skipped
  and reported once instead of ending the session; a NaN pose never reaches the filters or your VTuber app (last
  good pose used); a bad message from a page is ignored; Ctrl+C / any exit still closes camera, microphone and
  helpers; the WiLoR helper restarts itself (twice) if it dies and quits when the studio is gone; the camera reader
  survives driver errors and closes cleanly; lip sync audio can't fall behind (bounded queue); a busy viewer port
  is reported instead of a silent failure; closing the studio window with its X quits cleanly (it used to reopen);
  pressing C no longer forgets the K hand calibration; the viewer and settings page retry gently (up to every 5 s)
  while the studio is off; the WiLoR helper runs from a temp copy, because the Microsoft Store Python that
  `.venv_wilor` is built on can't see the one-file app's unpack folder (fingers silently fell back to MediaPipe).

**Start-up** (development run, launch → avatar data flowing): **2.4 s → 1.3 s**. The studio's imports went from
1.8 s to 0.55 s: MediaPipe's package start loaded its audio classifier (sounddevice) and drawing helpers
(matplotlib) that this app never uses - they're skipped now (`all_in_one._import_mediapipe`); the voice code's
resampling filter no longer imports `scipy.signal` (0.7 s; same filter to 1e-15 with numpy). The webcam opens, the
microphone starts and the GPU whole-body model loads on helper threads while the face/hand/body models load, so
nothing waits for them. The one-file app reads its version from the last 128 bytes of the file instead of the zip
directory, and matplotlib is left out of the build. The studio prints `ready: avatar data flowing N s after start`.

**Per frame**, also off the tracking loop now: drawing the studio window, cutting WiLoR's hand patches, encoding a
`--record` video (it also keeps real-time pace now, so a take always replays at the speed it was recorded).

**Tried and dropped - fast hand tracking:** MediaPipe's hand model searches the whole picture for palms every frame
it tracks fewer than two hands (~15 ms). Running only its landmark model on crops around each hand (MediaPipe's own
model, converted to ONNX; it reproduced MediaPipe to 0.01 px) cost 4 ms a hand on the CPU / 1.8 ms on the GPU - but
on a real take the whole frame got no faster (18.6 vs 18.9 ms; head-and-shoulders framing 14.2 vs 14.3 ms) because
the body model (13 ms) is the real limit, and it found fewer hands when they're close together. Running it on the
GPU next to the whole-body model also hung the GPU (two DirectML sessions at once), so that's avoided everywhere.
MediaPipe itself can't use the GPU on Windows (`NotImplementedError`), so its ~20 ms on the CPU is the floor for
this design; going lower means running MediaPipe's face, hand and body models on the GPU through ONNX with one
scheduler for all GPU work - a separate project.

**What latency is physically possible**: a 30 fps webcam delivers a picture every 33 ms and its exposure and USB
transfer add more (the clap test measures it - typically 40-70 ms on its own), so end-to-end motion → avatar can't
get near 1 ms with any software; the studio's own share is now ~27 ms, of which ~20 ms is MediaPipe on the CPU
(it has no GPU mode on Windows). Prediction (*settings: look-ahead*) can hide a few tens of ms at the cost of
overshoot and more shake.

## Latency work (measured live, camera on, headless)
| | fps | frame → avatar data (median / 95th) |
|---|---|---|
| this morning | 29 | 22 / 37 ms |
| Windows throttling the studio + dim light | 15 | 43 / 62 ms |
| **now** | **29.7** | **18–24 / 24–30 ms** |
- **Windows throttling** (`avatarstudio/core/perf.py`): on CPUs with fast and power-saving cores, Windows 11 slows down programs that
  aren't the focused window, and the studio almost never is (your VTuber app or game is). The studio now opts
  itself out and runs at above-normal priority: tracking 36 → 20 ms per frame. Nothing system-wide is changed;
  `--no-boost` turns it off.
- **Camera in dim light** (`avatarstudio/tracking/camera.py`): Windows' default camera interface halved the webcam to 15 fps in dim
  light (+33 ms delay). DirectShow kept 30 fps with the same auto-exposure, so it's tried first.
- **Camera auto-exposure is switched on at every start.** Webcams remember a manual exposure inside the device;
  an exposure test left this one at a fixed short exposure (picture brightness 15/255, nothing trackable). The
  studio now always turns automatic exposure back on (tested: forced dark → 110/255 at 28 fps on opening).
- **GPU whole-body model** no longer makes a frame wait: for left/right a result one frame old works as well.
  If the GPU drops out (laptop graphics switching, driver reset) it restarts the model once, then carries on
  with MediaPipe alone instead of crashing.
- Feeding MediaPipe smaller images made no difference (its models are fixed-size inside), so full resolution stays.
- The webcam can't go above 30 fps in any mode, so that's the floor for camera delay with this camera.
```bash
.venv\Scripts\python.exe tools/build_dist.py
```

## GPU whole-body model (RTMW)
`avatarstudio/tracking/wholebody.py` runs **RTMW-x** (OpenMMLab; 133 body, hand and face points, trained on 14 combined datasets) on
the GPU through ONNX Runtime + DirectML, in parallel with the MediaPipe models (~27 ms on an RTX 4060 while
another job was also using the GPU; 76 ms on the CPU). It is 2D only, so MediaPipe keeps providing depth.
- **On by default: left/right of each hand**, run only while left/right is in doubt (see *Speed and robustness*).
  Its wrists decide which hand is which. On a test video with hands
  near the face, where MediaPipe's body model loses the wrists, left/right label flips went from 744 per minute
  to **0**.
- Blending its finger/arm points into MediaPipe's was tried and removed: on your recording it made finger joints
  slightly *less* plausible (20.4° → 21.2° sideways bend). Left/right flips: 14.2 → **0** per minute with it.

## GPU tracking (optional, off by default)
`avatarstudio/tracking/gpu_tracking.py` runs MediaPipe's own body and face models (and its palm detector) on the GPU through ONNX Runtime +
DirectML - MediaPipe's own GPU mode isn't implemented on Windows - and the hand models on the CPU, two hands at once.
What MediaPipe does around its models is copied: the crop from the previous frame's points, turning the points back
into the picture, the body model's heatmap refinement, its video-mode smoothing. Switch it on with *GPU tracking* on
the settings page (from the next start) or `--gpu-tracking`.

Studio replay of take.mp4 on the RTX 4060 laptop (WiLoR and the whole-body model on unless noted):

| | tracking per frame | frame → avatar data (median / 95th) |
|---|---|---|
| CPU (MediaPipe), as fast as possible | 20.4 ms | 24 / 32 ms |
| **GPU tracking**, as fast as possible | **16.7 ms** | **18** / 35 ms |
| CPU, at 30 fps like a webcam | 20.1 ms | 24 / 33 ms |
| **GPU tracking**, at 30 fps | **16.8 ms** | **20** / 41 ms |
| GPU tracking, whole-body model off | 13.3 ms | 16 / 35 ms |
| GPU tracking, whole-body model and WiLoR off | 10.8 ms | 13 / 29 ms |

- **Same results as MediaPipe**: hands found 99.7% of MediaPipe's (frame by frame), left/right the same for 99.7–99.9%,
  face points 0.05 px apart, face shapes 0.001, body points ~2 px (the body model's own noise between two runs of
  the copy) with the same jitter. With the switch off nothing changes: the replay is identical to before it existed.
- **Costs**: about 1 s longer start (DirectML compiles its models and holds Python meanwhile; with the webcam this
  overlaps the camera opening), and the slowest frames got slower: when a new hand comes into view MediaPipe's own
  hand search still runs (~20 ms on the CPU, on ~12% of take.mp4's frames), and the whole-body model takes the GPU
  for 17–30 ms every 3rd frame, so frames in between run the same models on the CPU (~21 ms).
- **One GPU thread**: all DirectML work, the whole-body model included, goes through one thread - two DirectML
  sessions running at the same time on different threads hung the GPU. While WiLoR (CUDA) is busy every GPU run waits
  its turn, so body + face are one merged GPU run (separate runs took twice as long) and the hands stay on the CPU.
- The models (`models/mp_*.onnx`) are MediaPipe's own, converted by `tools/convert_landmark_models.py` (developer tool;
  `--check` compares them with the TensorFlow Lite interpreter: same outputs to ~0.001 px).

**Compare setups on your own recording** (every upgrade has to prove itself in numbers):
```bash
.venv\Scripts\python.exe app.py studio --record take.mp4
```
```bash
.venv\Scripts\python.exe tools/compare_tracking.py take.mp4
```
It reports ms/frame, hands found, left/right flips, finger-joint plausibility and fingertip jitter for MediaPipe
only and + GPU left/right.

## Personal face refiner
Launcher: **Train face** (or `scripts/windows/Train Face.bat` in the project folder; about 90 s): 20 prompted expressions (smile, brows up, wink left/right,
cheek puff, pucker, jaw open, ...). From them it learns, for each face shape, your resting level, your own
strongest version (→ 1.0), and **cross-talk**: shapes MediaPipe wrongly raises during *other* expressions on your
face, e.g. eyes half-closing in a big smile. It checks itself on half the recording it didn't train on and only
saves `face_refiner.json` if that is no worse. When present it replaces the generic resting-face subtraction,
auto-range and blink adaptation (switch: *Personal face refiner* on the settings page).

Tested on a simulated user only so far: asked-for expressions went from 0.74 to 0.91 of full strength and
cross-talk from 0.131 to 0.013; a big smile no longer half-closed the eyes (0.51 → 0.02) while a real wink still
registered (0.82 → 0.91).

**Tongue out, cheek puff, lip funnel** (`avatarstudio/personal/mouth_model.py`): MediaPipe has no tongue, and its cheekPuff /
mouthFunnel barely move for many people. Train Face now adds three prompts (tongue out twice, and talking
normally so speech isn't mistaken for them) and trains a small classifier on a level, tilt-corrected crop of your
mouth and cheeks. A shape is only switched on if it was recognised reliably on held-out moments of your recording
(≥ 60% found, ≥ 80% of detections right). Sent as ARKit `tongueOut` / `cheekPuff` / `mouthFunnel` (Perfect Sync)
and a custom VRM clip `TongueOut` if your model has one. Pipeline-tested with a synthetic tongue painted onto your
take: found 96%, 98% of detections right; real accuracy needs your recording.

## Check your VTuber app + model (do this first)
Launcher: **Test avatar setup** (or `app.py studio --test-pattern`). It sends 24 known steps (T-pose, arm
forward/up, elbow, shoulder shrug, palm up, fist, pointing, head turn/nod/tilt, lean, eyes, step forward, wink,
A/I/U/E/O, Joy, Perfect Sync smile and cheek puff), each held for 3 s with a description of what the avatar should
do. A step that looks wrong tells you exactly which mapping is off.

Verified so far with pixiv's VRM 1.0 sample model in the built-in viewer (three-vrm), by measuring the model's bones
for each step: all 22 checkable steps are correct. The 2 Perfect Sync steps need a Perfect Sync model. Inside
VSeeFace, Warudo, etc. it's still unverified; that's what this pattern is for.

## Arms
- **Shoulders** lift with the arm once it rises above 20° from horizontal (35% of the extra rise), like a real
  shoulder blade. The upper arm is solved relative to the lifted shoulder, so it still points exactly where yours does.
- **Forearm twist:** VRM models have no twist bone, so turning the wrist twisted only at the wrist and pinched the
  skin there (the "candy wrapper" look). Half of the twist now goes to the forearm; the hand stays exactly where it was.

## Eye gaze calibration
Press **G** (or *Eye gaze* on the settings page): a full-screen dot moves to the centre and four corners. Follow it
with your **eyes only**, head still. It fits your iris movement to the screen edges, so the avatar's eyes match
where you look. If your head moved instead of your eyes, the calibration is rejected with a message.

## Latency
- When the studio quits, it prints **frame arrival → avatar data sent**: the studio's own delay (median and 95th
  percentile).
- Launcher: **Measure latency (clap test)** (or `--latency-test`): clap slowly 6 times with both hands in view. The microphone hears
  each clap within about 10 ms and the camera shows it later; the difference, plus the studio's own delay, is your
  real motion-to-avatar delay. Tested with a simulated 120 ms camera delay, it measured within one frame (33 ms).

## Long sessions
- If the **camera** disappears (unplugged, taken by another app), the avatar eases to rest and the studio reopens the
  camera every second; tracking resumes on its own. Tested with a simulated camera that died mid-stream.
- If the **microphone** stream dies, it is reopened every 2 s (re-plugged devices are found again).
- `--stats-every 60` prints fps, memory, pipeline delay and reconnect counts every minute.

## Settings page, profiles, hotkeys
Everything below can be changed live at **http://127.0.0.1:8000/settings.html**, no restart needed:
expressiveness, auto-range, linked blinks, microphone lip sync and its weight, smoothing, prediction, idle
motion, root motion, mirror, frame-rate cap, VTube Studio output and what the studio window shows. It also has
buttons for calibration and expressions, and live status (fps, face/hands, calibration, mic level, VTube Studio).
- Changes are saved in `settings.json`. Command-line flags only apply to that run and are never saved.
- **Profiles** store settings *and* calibration under a name (e.g. "desk", "couch"), in `profiles/`.
- **Expression hotkeys** live in `hotkeys.json` (key → expression clip, toggled on/off). Keys 1–9 work in the studio
  window; **Ctrl+Alt+key** works while another window (a game, OBS) is focused. Only those combinations are
  listened for.
- **Frame-rate cap** (e.g. 20 fps) leaves CPU for a game while streaming.

## Microphone lip sync
Your voice drives the mouth together with the camera (`avatarstudio/audio/mic.py`):
- **Loudness** says how far the mouth opens. The background is measured continuously (the quietest 15% of the last
  6 s), so room noise doesn't open the mouth.
- **Which vowel:** the two main resonances of your voice (formants), estimated about 30 times a second.
  The V calibration records *your* voice; until then, textbook averages are used, which picked 13 of 15
  synthetic test vowels correctly ("I" is the hardest).
- **Blending:** while you speak, the voice decides how far the mouth opens and shares the vowel choice with the
  camera (*voice share* setting, default 50%). It keeps working when your face turns away.
- **Silence gate:** when the mic hears nothing, vowels only show if your lips are really apart. While you're silent
  with your lips together, the tracker also re-learns your resting lip rounding, because MediaPipe's reading of a
  relaxed mouth drifted between sessions (0.33 → 0.44 → 0.60) and made a closed mouth look like "U". Replaying a
  recorded session, frames where a closed, silent mouth showed a vowel went from 43% to 0%, and speech got
  slightly stronger (median 0.56 → 0.66).
- `--no-mic` turns it off; `--mic-device N` picks another microphone (or choose it in the launcher).

**Speech detection** (on by default, `avatarstudio/audio/voice.py`): Silero VAD (MIT license, 2 MB) decides every 32 ms whether the
sound is a human voice; only then may the voice open the mouth. **Noise suppression** (off by default): RNNoise
(BSD license) cleans the voice before lip sync; microphones running at 44.1 kHz are converted to its 48 kHz.
Both are switches on the settings page. Tested offline with Windows text-to-speech voices mixed with synthetic
fan noise, typing and music:

| noise only: mouth opens | plain | speech detection | noise suppression | both |
|---|---|---|---|---|
| fan | 0% | 0% | 62% | 0% |
| typing | 78% | **0%** | 85% | 0% |
| music | 14% | **0%** | 46% | 0% |
| fan + typing | 45% | **0%** | 91% | 0% |

| speech: caught / vowel agrees with clean | plain | speech detection | noise suppression | both |
|---|---|---|---|---|
| clean | 100% / 100% | 98% / 100% | 100% / 64% | 97% / 64% |
| loud fan (SNR 3 dB) | 60% / 58% | 58% / 61% | 99% / 51% | 97% / 50% |

Speech detection removes false talking at almost no cost. Noise suppression rescues mouth opening in a loud room
but changes the voice enough that vowel guesses agree less, hence a switch for noisy rooms. The latency-test clap
detector always listens to the raw microphone.

## Less jitter
Measured on `take.mp4` (38 s) by replaying the tracking through the rig. Shake = average turn per frame while
a body part is held still; glitches = jumps > 10° in one frame that a smooth (lag-free, offline) reference
doesn't have.

| | body shake | finger shake | finger glitches |
|---|---|---|---|
| previous build | 0.288° | 0.557° | 137 |
| now | 0.241° | 0.451° | 63 |

- **Fingers are solved as real fingers move** (`rig._finger_set`): knuckle bend + a little spread, middle and
  tip joints pure hinges, no twist; the tip joint follows ⅔ of the middle joint. Solving each joint as a free 3D
  rotation turned tracking noise into twist flips and limit-clamp jumps.
- **Prediction now defaults to 0 ms** (settings page): extrapolating also extrapolates noise (it was the largest
  single source of glitches). Finger smoothing a little stronger (3 → 2 Hz).
- **Hands that appear or disappear ease in/out** (0.2 s / 1 s) instead of snapping between tracked and relaxed.
- **3D viewer glides between poses:** the studio sends ~30 poses a second, the screen draws 60+, so setting each
  pose directly made the avatar step. Each drawn frame now moves toward the latest pose (30 ms time constant).
- Gesture snapping (preset hand poses) was removed: the hand-made presets looked wrong, and there's no free,
  licensed set of VRM hand-pose presets to download instead.

## Fists, thumbs up, peace: finger shapes
What the tracker reads on your `take.mp4`: a fist's knuckles at ~41-48° (real ~90°) and middle joints at 41-69°
(real ~100°), and a thumb hidden in a fist drawn almost straight. So on the avatar fists stayed half open with the
thumb sticking out, and open hands looked like claws (old solver). Now (`rig._finger_set`):
- **The thumb** has its own anatomical model: a base that swings across the palm and tilts toward it, then two
  hinges, with human limits (it used to be a free 3D rotation).
- **Your bend range is stretched**: what you read with an open hand → straight, with a fist → fully curled
  (default range measured on your take; gain capped so noise isn't blown up).
- **Knuckles follow the middle joints** as a finger curls (real fingers flex together).
- **A hidden thumb is folded** when the ring/little fingers are curled and the thumb tip is near the index
  knuckle. On your take: folded in 58% of fist frames, 79% pointing, 83% peace, **0%** thumbs-up / open / OK.
- **Neighbouring fingers can't cross** sideways through each other.
- Checked by rendering your real fist / thumbs-up / pointing / peace / OK / open frames on your VRM, old vs new;
  single-frame finger glitches stay at the old level (75) with less shake.
- **K** (or *Hands: open + fist* on the settings page): 3 s open hands flat, 3 s tight fists → your own range,
  saved in your calibration.

## WiLoR fingers (3D hand model, GPU)
[WiLoR](https://github.com/rolpotamias/WiLoR) (CVPR 2025) reconstructs a full 3D hand (MANO) and stays plausible
when fingers hide each other. It gives the **finger shapes**; the wrist and hand orientation stay with MediaPipe's
newest frame, so WiLoR's delay only affects the fingers (`avatarstudio/tracking/wilor_client.py`, setting *WiLoR fingers*).
It needs PyTorch, so it runs as a helper process (`avatarstudio/tracking/wilor_server.py`) in its own environment, `.venv_wilor`; the
studio and packaged builds stay PyTorch-free (a build inside `Distributions/` finds the project's `.venv_wilor`).
Until it's ready (~12 s after start) or if it falls behind, fingers come from MediaPipe.

Measured on `take.mp4` with the same rig:

| fingers | MediaPipe | WiLoR (refreshed every 2nd frame) |
|---|---|---|
| shake while still | 0.484° | 0.371° |
| single-frame glitches | 78 | 55 |
| wrist glitches | 6 | 0 |

It reads open hands straighter (fingertip joints ~1° instead of 6-13°) and fists more curled. Live on the RTX 4060
laptop (with the GPU body model also running): results arrive **71 ms** after the frame (95th 91 ms), 97% of
hand-frames use them, studio at full speed. Speed-ups: the helper opts out of Windows throttling (`avatarstudio/core/perf.py`) and
prepares both hands with OpenCV in one batch instead of WiLoR-mini's per-hand full-image blur (same points: median
difference 0.6 mm); 168 → 71 ms.

Setup. WiLoR fingers are optional and nothing of them comes with Avatar Studio: you install them yourself and accept
their licences yourself. WiLoR's weights are non-commercial (CC BY-NC-ND 4.0), the MANO hand model is licensed for
non-commercial research, education and artistic projects only, and the hand detector (ultralytics) is AGPL-3.0
([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
```bash
python -m venv .venv_wilor
```
```bash
.venv_wilor\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```
Then install WiLoR-mini from its source archive (github.com/warmshao/WiLoR-mini) with `torch<=2.5` loosened to
`torch>=2.0` and the `chumpy` line removed from its `requirements.txt`, plus `pip install dill`.
Get the MANO hand model from its makers: register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de), accept its
licence, download *Models & Code* and copy `mano_v1_2/models/MANO_RIGHT.pkl` into
`.venv_wilor\Lib\site-packages\wilor_mini\pretrained_models\` **before** the first run (WiLoR-mini downloads a copy
of its own only when that file is missing). Then
```bash
.venv_wilor\Scripts\python.exe tools/wilor_fix_mano.py .venv_wilor\Lib\site-packages\wilor_mini\pretrained_models\MANO_RIGHT.pkl
```
rewrites it without the old `chumpy` library, which no longer installs (it keeps a backup). The first run downloads
WiLoR's weights and hand detector from WiLoR-mini's Hugging Face page. `tools/wilor_bench.py` measures speed and
saves its poses for a recording.

## Thumb, wrist limits, hidden parts, shrugs
- **Thumb bends across the palm, not into it.** The thumb is turned about its own axis compared with the fingers,
  so its outer joints bend toward the little finger. Measured on WiLoR's hands of your take: the real bend axis
  is turned **62°** from "toward the palm" (1109 bent thumbs, middle half 54-69°). With that axis, the rebuilt
  thumb tip matches your real thumb within **4°** (median; 12° for the worst 10%) instead of 25° (46°).
  The "fold a hidden thumb" fix only runs for MediaPipe fingers; WiLoR reconstructs hidden thumbs itself.
- **Wrist limits.** The solved wrist went past human range in 30-40% of hand frames (sideways up to 138°), mostly
  where the body and hand trackers disagree about the forearm. Now kept within bend -70..80°, sideways -35..25°,
  twist ±90° (`rig.limit_wrist`), checked again on the final pose (after holds, no-clip and fingertip contacts),
  so the wrist the avatar's skin shows never goes past them on your take. Elbows and forearms were already in range.
- **Hidden by a hand = kept, not guessed** (`rig._occlusion`):
  - a hand over your mouth (3+ hand points over the lips) or eyes: that region's face shapes, the vowels and the
    expression keep their last uncovered values; your microphone still moves the mouth;
  - face fully hidden behind a hand: head pose and expression are held up to 10 s (was 0.3 s, then neutral);
  - a hand out of sight near your head (behind it, rubbing it): arm and hand pose held up to 10 s. The body
    model's wrist estimate is used to tell this from a hand dropping out of view (that one still relaxes).
- **Shoulder shrug** is tracked: how far each shoulder rises toward its ear (ears barely move when you nod), against
  your relaxed level; the larger of that and the automatic raised-arm lift. Shoulders forward/back is not tracked:
  the chest direction itself comes from the shoulders, so it would look like a torso twist, and webcam depth is too
  noisy to separate them honestly.
- **Up / down** (sitting up, slouching, crouching): the shoulder line's height vs. your calibration, minus the drop
  a forward lean already shows. Needs the C calibration done once with this version.
- **The back bends over two joints** (spine + chest, half each) instead of one hinge: smoother bends, and body shake
  0.241 → 0.172°/frame.

## Fingertips that touch, touch on the avatar
Your fingers are tracked fine, but the avatar's fingers have other lengths than yours, so a real pinch used to
leave a gap (median 6.6 cm on your take). Now (`rig.solve_pinch`, `rig.solve_two_hand`):
- **thumb to a fingertip** (OK sign, pinch): when your thumb tip is within ~3 cm of a fingertip, the avatar's thumb
  and that finger are solved together, inside human joint limits and as close to your pose as possible, so the tips
  meet. Gap on your take: **6.6 → 0.0 cm** (90% of pinch frames under 1 mm), in ~1 ms per pinching hand.
- **fingertips of both hands** (steepled fingers, tapping fingertips): both arms move a little so the touching tips
  meet halfway. Gap **9.8 → 2.3 cm** median (wrist limits win where the two disagree).
- It eases in with how close the tips are, so near-touches don't snap. Always on; it's applied before output, so VMC apps get it too.

## Hands on your face, head turned away
- **A hand on your face lands on the avatar's face** (`rig.face_contact`, `rig.solve_face_contact`). The body
  model guesses depth badly for a hand at your face, and the no-clipping sphere around the head reaches further forward
  than a face, so the avatar's hand used to hover about **33 cm** in front of its face (median, your take). Now each hand
  point on your face (fingertips, middle of the palm) is placed on the same spot of the avatar's face: the face mesh point
  under it plus its offset in the picture, turned into centimetres from your pupil gap, carried over in your head's
  frame, and scaled to the avatar's eye spacing. The arm is then solved to put it there. At the face the arm is folded
  tight, so every move of the wrist turns the hand a lot; the solver learns how the touch point follows the wrist
  (Broyden), and that takes ~5 passes where plain repeats took 8 or more. Gap: **33 → 1.2 cm** (90th percentile 4.3).
- **Touching vs. in front of the face:** hands that only cover the face in the picture (arms crossed in front of it)
  look bigger than hands that touch it. On your take, the largest palm segment ÷ pupil gap reads 1.11 for a hand on the
  face and 1.41 for arms crossed in front of it. The distance estimates couldn't tell the two apart: they put hands on
  the face anywhere from 52 cm in front to 59 cm behind it. Crossed arms are now left alone in 69% of frames.
- The touch eases in over ~0.1 s, out over ~0.15 s, and holds through a hand lost for a frame or two (or hidden behind
  the face), so hands don't snap on and off the face. Per frame it adds ~1.3 ms to the rig on a take where your hands are
  at your face most of the time (2.2 → 3.5 ms), and nothing when they aren't. Off: `rig.face_contacts = False`.
- **Head turned far away (side-on, ~90°):** the face mesh drops out, and the head used to freeze, then fade to neutral.
  Now it carries on from the body model (nose and ears): the last face-tracked pose plus how far the body model's head
  has turned since. While the face is seen the face mesh stays in charge, because the body model's head direction is
  ~9° off it and shakes more (0.94 vs. 0.61°/frame). Switching between the two fades over 0.25 s. A hand covering your
  face still just holds the head. Your take has no full side-on turns, so record one (turn fully left / right, look up /
  down) to check it.

## Avatar proportions
The 3D viewer sends your loaded VRM's skeleton to the studio, which then uses those bone lengths (instead of
average proportions) for no-clipping and fingertip contacts. Load your VRM in the viewer once per session; the
studio log shows `avatar proportions received`. VSeeFace / Warudo / VNyan don't share skeletons, so with those the
average proportions are used (load the same VRM in the viewer too if you want exact contacts there).

## Torso and shoulders
- **Body model "full"** (`--pose lite` is faster, less accurate). The larger "heavy" model's torso turn agreed
  better with the head turn on your take (correlation 0.71 vs 0.54) but doubled tracking time, so it was dropped.
- **Torso turn and side tilt** come from the shoulder line; **forward / back lean** from the shoulders vs. hips, and
  when you're sitting and your hips are out of view, from how much closer your shoulders come to the camera than
  at calibration (seated lean). The avatar then leans instead of sliding toward the camera.
- **Shrugs** per shoulder (see above). The bend is shared by spine and chest.
- Not possible from one webcam: rolling the shoulders forward/back separately from a torso twist.

## Two or three cameras (experimental)
Add a 2nd and 3rd webcam, e.g. one left and one right of the main (centre) one: launcher → *Extra cameras*, or
`app.py studio --camera2 1 --camera3 2`. Then press **C** (neutral pose) once with all cameras seeing your upper body:
that lines them up (`avatarstudio/tracking/multicam.py`). No checkerboard is needed: MediaPipe's 3D body points are centred on your hips in
every camera, so cameras differ only by a rotation, which is fitted to the body points both see clearly. A camera
that can't be lined up (doesn't see enough of you, or its picture is mirrored) is reported and not used. The body
model swaps left and right on a mirrored picture, so C (holding still) rarely notices one: turn mirroring off in
the camera's settings. Recorded takes check it properly, see *With two or three cameras* below.
- **Body:** every body point is averaged over the cameras that see it, weighted by how clearly they do - an arm
  hidden behind you from the front, or a hand turned away, comes from a side camera.
- **Hands:** a hand seen by several cameras is averaged (palm-on or back-on views count more than edge-on ones);
  a hand only a side camera sees is added.
- **Head:** turned away from the main camera, a side camera keeps your head and face tracked.
- The main camera keeps everything measured in its picture (your position, distance, the studio window, WiLoR's
  finger crops); the extra cameras show as small views in the studio window.
- Tested with recordings standing in for cameras (`--input2`, `--input3`) and with a simulated side camera 40° off:
  it lined up to within 0.33°, and arm points the main camera lost (guessed 30 cm off) came back to within 1 cm.
  **Not tested with real extra webcams yet.** Each camera adds its own tracker (~6 ms more per frame with two on
  this laptop; an extra camera's face model only runs while the main camera has lost your face); cameras aren't
  frame-synchronised, so fast moves can differ by a frame.
- **A phone as a camera** (e.g. Windows 11 *Settings → Bluetooth & devices → Mobile devices → use as a connected
  camera*, or an app like DroidCam / Camo): it shows up as an ordinary webcam. Its depth / laser-autofocus sensors
  are not passed on (only the picture), so it's a regular camera to the tracker - but often a better one (cleaner
  picture in dim light, 60 fps), and an easy 2nd camera. Use USB rather than Wi-Fi (less delay), and turn off
  stabilisation, beauty filters and auto-framing in the phone app: they move and reshape the picture, which the
  tracker reads as you moving. Compare with the launcher's *Measure latency* test.

## VSeeFace-style extras
- **Linked blinks**, **pause (P / Ctrl+Alt+P)**, **recalibrate from any window (Ctrl+Alt+C)**.
- **Body follows head:** when your shoulders are out of view (close framing), the upper body turns and tilts with
  20% of the head's motion instead of standing stiff; the head itself stays exactly as tracked.
- **Body look-ahead** (settings page, default 10 ms): torso, head and arms are predicted slightly ahead to hide
  delay, fingers and face are not. On your take: body lag −17% for +8% shake; fingers unchanged (look-ahead on
  everything doubled finger glitches).
- **60 fps camera mode** (launcher or `--fps 60`), for webcams that can: halves the camera's share of the delay.
- **Launcher:** one Start/Stop button with a running / not-tracking line; a *Getting started* checklist that ticks
  off what's done (neutral pose, vowels, hands, gaze, face) and spells out the next step; buttons grouped as Open /
  Check / Personalize with a hover tip on each; start tracking automatically when it opens; minimize to the tray
  (start / stop / viewer / settings from the tray menu); Start menu + desktop shortcut; WiLoR installed or not.
- **Settings page:** *Reset all settings to defaults* (calibration is kept), extra-camera and WiLoR status.
- **Closing the studio window** with its X quits cleanly (it used to reopen).

## No clipping through the body
After the arm pose is solved, the elbow, wrist and middle of each hand are checked against a torso capsule and a
head sphere; anything inside is moved out with a 2-bone arm solve (`rig.keep_arm_out_of_body`), so a hand rests on
the chest / face instead of sinking in. It's applied before output, so VSeeFace / Warudo / VNyan get it too.
Test: a wrist forced 5.1 cm into the chest ends up on the surface. The body volume uses average proportions;
*Body thickness* on the settings page adjusts it for bulky avatars or clothes (*No clipping* switches it off).
Hair / cloth passing through the body is the avatar's own spring-bone setup, not tracking.

## VRM 0.x models in the 3D viewer
Older **VRM 0.x** models (e.g. exported with the Blender VRM add-on) showed up from behind with
the arms raised, while VRM 1.0 models were fine. Both were viewer bugs: VRM 0.x models face the other way and
their bone axes are mirrored, and the viewer undid its own turn-around and didn't flip the rotations. Fixed;
both of your models now give the same poses (measured: arms down, left arm forward, right arm up).
VSeeFace / Warudo / VNyan handle VRM 0.x themselves, so VMC output was not affected.

## Smoothing tuned to you
```bash
.venv\Scripts\python.exe app.py tune-smoothing take.mp4
```
(or *Tune smoothing from take* in the launcher). The live filter can only look at the past, so it trades shaking
against lag. Offline, smoothing the recording forwards and backwards gives a shake-free reference without lag. For
each bone group it keeps the setting that comes closest to that reference **without shaking more than the
default when you hold still**, and saves `smoothing.json` (loaded automatically; delete it to go back).
On your take: body and hands **24–25% closer** to the reference (less lag) with the same stillness; fingers and
eyes were already as good, so they keep the defaults.

## Tracking loss, idle motion, prediction
- When your face, torso or an arm drops out, its last pose is held for 0.3 s, then eased into a rest pose over
  1 s (no snapping). Face shapes fade to neutral; VTuber apps otherwise keep the last value, e.g. an open mouth.
- **Idle:** gentle breathing always; while your face is lost, automatic blinks and a slow head drift.
- **Camera covered** (a lens cover, a hand over it: the picture is black, with not even a lamp or screen showing):
  after 0.4 s the picture isn't tracked at all, the avatar eases to rest and its idle motion fades out over a
  second, so it stands still. The status bar shows *camera covered*; the microphone still moves the mouth. A dim
  room isn't mistaken for it (tested: a photo at a quarter of its brightness). Tracking resumes on the first lit frame.
- **Prediction** (default 0 ms; settings page) extrapolates movement to hide camera and processing delay. In a test
  with fast 1 Hz motion, 20 ms removed a 33 ms lag at the cost of an 8.5% overshoot, but it also makes noise
  visible as jitter, so it's off by default.

## Your VRM in the 3D viewer
Open the 3D viewer (launcher button, or http://127.0.0.1:8000), click **Load VRM…** or drop a `.vrm` file
(VRM 0.x and 1.0, via [three-vrm](https://github.com/pixiv/three-vrm)). Bones, root motion, VRM expressions and,
for Perfect Sync models, the ARKit shapes are applied live, including hair and cloth physics. Tested with pixiv's
VRM 1.0 sample model: it faced the camera and reproduced a crossed-arms pose with each arm on the correct side.

## Everything in one window
`AvatarStudio.exe` (development: `python app.py`) opens one app window (`avatarstudio/ui/mainwindow.py`, `avatarstudio/ui/web/app.html`):
- **Live:** the camera picture with what's tracked, and your 3D avatar, side by side (or only one of them). Below
  them: the calibration buttons (Neutral pose C, Vowels V, Hands K, Eye gaze G, Pause P), the expressions, and a
  status bar. Prompts like "Say A now" appear over the page; the eye-gaze dots fill the screen.
- **Settings** (they work while tracking is off too), **Personalize** (face training, take + smoothing, custom
  gestures), **Motion capture** (record, mocap a video, results), **Check setup** (avatar test, latency test),
  **Getting started** (the checklist, with a button for each step).
- **Tools inside the window:** face training and gesture recording run on the Live page like tracking (your
  camera with each prompt and its progress over the page; *Cancel* stops them), then the app goes back to
  Personalize with the result. Custom gestures: record each sign (and "none") with a button, *Train*, and pick the
  expression each sign shows from a list (gestures/actions.json is written for you). Motion capture and smoothing
  show a progress bar (tracking frames, filtering, drawing the overlay video) and their messages.
- Camera and microphone are in the top bar; the frame rate, extra cameras, start-with-the-app and the tray are
  under Settings. Keys work while the window is in front, Ctrl+Alt ones from anywhere as before.

Tracking still runs as its own process (`app.py studio --ui`), now without a window of its own: its picture comes into
the app as a JPEG stream (`http://127.0.0.1:8000/camera.mjpg`, only while shown), its data over the same
WebSocket as before. A separate process keeps the window responsive whatever tracking does; VMC and the OBS page
(`http://127.0.0.1:8000`) work as before. Measured with the `take.mp4` replay: window ready 1.6 s after start,
camera picture 0.85 s after *Start tracking*, tracking cost the same with and without the picture (its drawing
and JPEG encoding run off the tracking loop). Face training and gesture recording use the same picture stream
(`app.py train-face --ui`, `app.py gestures record NAME --ui`; preview.ToolView); run on their own they still open their
OpenCV window as before. `AvatarStudio.exe launcher` opens the classic launcher, which is also what opens where no
web view is available.

## The 3D viewer and settings in their own windows
The launcher's *3D viewer* and *Settings page* buttons (and the tray menu) open them as Avatar Studio windows
instead of browser tabs: Microsoft Edge WebView2 through pywebview (`avatarstudio/ui/webwindow.py`), also as
`AvatarStudio.exe viewer` / `AvatarStudio.exe settings` (development: `python app.py viewer`). They are the same
pages, talking to the studio over the same WebSocket. Each window serves the pages itself, so it can open before
tracking starts and connects once the studio runs. Size, position and maximized state are remembered
(`windows.json` in your data folder; the first time it opens centred on the screen with the mouse). Opening one
that is already open brings it to the front. Its *Settings* / *open 3D viewer* links open the other window.
- **No internet needed:** three.js 0.160.0 and three-vrm 3.4.0 are bundled in `avatarstudio/ui/web/lib/` (MIT licences next to
  them), for the windows, browsers and OBS alike. `avatarstudio/ui/web/index.html` opened straight from the folder
  (`file://`) loads them from unpkg, and can't connect to the studio (only pages served on this PC's address can,
  see *Only your own pages can use the studio*). Checked with the window's internet blocked: the viewer loaded,
  connected, took a dropped VRM and moved it.
- **Browsers and OBS:** http://127.0.0.1:8000 and http://127.0.0.1:8000/settings.html work as before while the
  studio runs.
- **OBS mode:** http://127.0.0.1:8000/?obs is the viewer with nothing but the avatar: no panel, grid or tracked
  points, a transparent background (OBS browser sources keep it), and the camera framed on the avatar's head height
  once it loads. `&bg=00ff00` (any hex colour) gives a colour-key background instead, `&view=full` / `upper` /
  `face` the framing. The avatar is the `.vrm` picked in Settings → OBS: the app copies it to `obs_avatar.vrm` in your
  data folder and the studio serves it as `/avatar.vrm` (same local-pages-only checks as the rest). Without one a
  skeleton shows. OBS loads the page once, so start tracking first, or press *Refresh* in the source afterwards.
- **Without WebView2** (it comes with Windows 11 and current Windows 10) or pywebview, the buttons open the browser
  as before. A WebView2 that fails to start also falls back to the browser.
- **Startup** (this laptop, one-file app): the window is on screen 0.7–0.8 s after the click, and the page is
  loaded and connected to the studio at 1.2–1.35 s. Most of that is loading .NET and WebView2. WebView2 keeps its
  cache in `%LOCALAPPDATA%\AvatarStudio\WebView2`. The one-file app grew by 1.7 MB (pywebview, pythonnet, the
  bundled JS).

## Only your own pages can use the studio
The studio's addresses (pages on http://127.0.0.1:8000, the camera picture, the WebSocket on :8765) listen on this
PC only, but a browser lets any website reach 127.0.0.1. So they now answer only Avatar Studio's own pages
(`avatarstudio/core/localweb.py`): a browser request has to come from a page on 127.0.0.1, localhost or [::1], the
address it asks for has to be local (no DNS-rebinding tricks), and other sites can't frame the pages or load the
camera picture. Programs that aren't browsers (the app itself, your own scripts) connect as before. Checked: from
https://example.com the WebSocket is refused and the camera picture doesn't load; the viewer at
http://127.0.0.1:8000 and http://localhost:8000 still connect and show it. Also: the viewer's and settings page's
`?ws=` can only name a local address, *Open data folder* only opens folders inside your data folder, and updates
download only from this project's own GitHub releases. Messages from pages are checked too: avatar bone lengths
that aren't real numbers (or are absurdly long) are ignored, expressions can only be ones from `hotkeys.json`, and
the WiLoR helper closes a connection that asks for more than 8 hands or patches over 2048 px.
- **The app windows show only their own pages.** What a page can ask of its window (uninstall, tools, updates)
  would be there for any page it showed: a link or file dropped onto a window is now refused, and a window that
  ends up on another page anyway goes straight back to its own (tested with https://example.com).
- **Gesture models are plain numbers now.** `gestures/model.joblib` was a pickle, which runs code when it's loaded,
  so a gesture model someone sent you could have taken over your PC. Training now saves `gestures/model.npz` (the
  network's weights only, read with pickles switched off) and the studio runs the network itself in NumPy (same
  probabilities as scikit-learn to 1e-8). An old `model.joblib` is never loaded: press *Train* once again, the
  recordings are kept. Every file the app reads is now data only (JSON, NumPy arrays, ONNX models).

## VTube Studio (Live2D)
1. In VTube Studio: Settings → turn on **Start API** (port 8001), and turn **off** its own webcam tracking.
2. Start the studio with `--vts`, or switch on *VTube Studio* in the settings page.
3. The first time, VTube Studio asks to allow **"Avatar Studio Tracker"**: click Allow. The token is saved in
   `vts_token.json`, so later starts connect automatically.

The tracker injects VTube Studio's standard parameters (FaceAngleX/Y/Z, FacePositionX/Z, MouthOpen, MouthSmile,
MouthX, EyeOpenLeft/Right, EyeLeft/RightX/Y, Brows, BrowLeft/RightY, CheekPuff, FaceAngry), so models set up for
normal tracking work. **Untested against the real VTube Studio:** it was tested only against a stand-in that
follows the documented API (authentication, token, injection). If a direction is reversed, flip that parameter
in your model's VTube Studio settings.

## Face and lip sync
- **Mouth vowels (A/I/U/E/O)** are measured from the lips themselves: inner-lip opening, mouth width and rounding,
  scaled by eye-corner distance so leaning in or turning doesn't change them. MediaPipe's own `jawOpen` misses a lot of
  mouth movement (lips clearly apart in a grin read only 0.11).
  - With the **V** calibration, your mouth is matched against your own five vowel shapes. Each vowel is a path
    from your closed mouth (0) to your recorded vowel (1). In tests: closed mouth = exactly 0, each vowel ≈ 1.0,
    half-way ≈ 0.5, and 99% of noisy vowel shapes were recognised correctly.
  - Without it, general rules for opening, width and rounding are used. 30% of the eye-corner distance counts as a
    fully open "A". With the earlier 45%, a live session never got past A = 0.4. The stronger range is less
    reliable on noisy mouth shapes, recognizing 80% of test vowels correctly versus 99% with the V calibration.
- **Blinks** adapt to you per eye: the tracker learns your open-eye level and how strongly your blinks register.
  Replaying a recorded session, fixed thresholds caught 6 blinks a minute; the adaptive version caught 19, within
  the normal human range of 15–20.
- **Auto-range:** each face shape is stretched so *your* strongest version reaches 1.0. It starts with no boost,
  learns your range over a minute or two, and is capped at 2×; blinks and gaze are left alone.
  `--no-auto-range` turns it off.
- **`--expressiveness 1.3`** exaggerates every face shape by 30% (0.8 = subtler).
- Mouth shapes get lighter smoothing than the rest of the face, so fast speech isn't blurred.
- **Linked blinks** (VSeeFace's "link eye blinking"): both eyes blink together unless you clearly wink. With the
  head turned, the eye farther from the camera is seen edge-on and reads as closing, so from 10° to 25° of turn
  it follows the near eye. On your take: lopsided blink frames 36 → 15, head-turn "winks" 123 → 49.
- **Expressions** (Joy, Angry, Sorrow, Surprised, Fun) come from hotkeys (1-9, Ctrl+Alt+1-9), the settings page
  or trained gestures. The automatic expression model was removed (it cost 44 MB and ~1 ms per frame and
  misread talking as surprise); Perfect Sync models get your real smile, brows and cheeks from the face shapes.
- **Face monitor (F):** a zoomed view of your mouth plus live bars of the shapes being sent, for tuning.
- For **Perfect Sync** models, the 51 ARKit shapes (lips, cheeks, brows, eyes) give the most expressive result.
  VRM-preset models get blink, A/I/U/E/O, look up/down and your hotkey expressions.

| Option | |
|---|---|
| `--mirror-avatar` | raise your right hand → avatar raises its left (mirror behaviour) |
| `--no-root-motion` | keep the avatar in place when you lean or step forward, back or sideways |
| `--vmc host:port` | VMC receiver address (default `127.0.0.1:39539`); `--no-vmc` to turn it off |
| `--record take.mp4` | save the raw webcam video |
| `--input take.mp4` | replay a recording (or a still image) through the whole pipeline |
| `--bvh take.bvh` | save the motion as BVH when you quit (import in Blender: File → Import → BVH) |
| `--log take.jsonl` | save every frame's tracking + avatar data |
| `--fov 65 --ipd 6.3` | your webcam's horizontal field of view and your pupil distance (cm) for this run (normally the settings, Settings → Camera and distance) |
| `--full-body` | also track legs (default: torso, arms, hands and head) |
| `--resolution 640x480` | lower webcam resolution for slower PCs (default 1280x720) |
| `--fps 60` | ask the webcam for 60 fps (halves its share of the delay, if it can; it says if it can't) |
| `--camera2 1 --camera3 2` | experimental extra cameras (see *Two or three cameras*) |
| `--expressiveness 1.3` | exaggerate all face shapes (0.8 = subtler) |
| `--no-auto-range` | don't stretch face shapes to your own range |

## Speed
About **30 fps live**, which is the webcam's own limit, on an i7-13620H + RTX 4060 laptop. When you quit, the studio
prints where each frame's time went. Measured live (headless, mic on):
`tracking 20.9 ms, rig 2.9 ms, send 0.7 ms, camera wait 9.2 ms`; frame → avatar data 24 ms median. See also
*Latency work* above.

What makes it fast:
- The face, hand and body models run **in parallel threads** (34 → 16 ms). The hand model is the slowest, so the
  lighter body model would gain nothing; the accurate one stays.
- The camera is read on its own thread, so tracking never waits for a frame it could already have.
- VMC messages are encoded directly (python-osc's builder re-parses each message: 7 → 0.7 ms per frame).
- **P / Ctrl+Alt+P pauses**: the avatar holds its pose and tracking stops using the CPU (e.g. while gaming).

If it gets slow again, check what else is using the CPU or webcam. A VTuber app's **own** webcam tracking
running at the same time will compete with this one, so turn it off and use the VMC receiver only.

## What gets sent to the avatar
- **Bones (VRM humanoid, local rotations, T-pose = identity):** Hips, Spine, Neck, Head, both eyes, upper/lower arms,
  hands and all 15 finger bones per hand (plus upper/lower legs with `--full-body`). Arms that are out of view rest
  down by the sides.
- **Root position:** stepping closer, back or sideways moves the avatar, measured from your calibrated spot.
- **Face, VRM presets:** Blink, Blink_L/R, A/I/U/E/O (from lip geometry, see below), LookUp/Down,
  and the expressions you switch on (hotkeys, settings page, gestures).
- **Face, Perfect Sync:** the 51 ARKit blendshapes MediaPipe provides (all except tongueOut), sent under both
  `eyeBlinkLeft` and `EyeBlinkLeft` spellings so either naming convention works.
- Everything is smoothed with a One Euro filter (steady when still, quick when moving).

## Hands and fingers
- **Palm or back of the hand:** each hand reports `facing` (palm / side / back toward the camera) and `palm_angle`
  (0° = palm squarely toward the camera, 180° = back of the hand), from the palm's 3D surface direction. The avatar's
  hand twist uses the same palm direction; on the sample photos the two agree within 5°.
- **Hands crossing:** the hand model guesses left/right from the hand's look and can swap or duplicate the label when
  hands cross. Each hand is matched to the nearest **body wrist** instead (the body model knows which arm is which),
  then to where each hand was in the previous frame. `side_from` says which rule decided.
- **Short dropouts:** a hand that vanishes for a moment (common when hands overlap) keeps its last pose for 0.5 s
  instead of snapping to a relaxed hand.
- **Joint limits:** finger joints are kept inside human ranges (no bending backwards past 25° or sideways at the hinge
  joints). On a tightly gripping hand, MediaPipe's raw 3D bent finger joints 33–72° sideways, which a real finger
  can't do; the limits replace that with the nearest possible pose.
- **Limits of one webcam:** fingers that are hidden behind other fingers or the other hand are *guessed* by the hand
  model. The steps above keep those guesses plausible and stable, but tightly interlocked fingers can't be
  measured from a single viewpoint. A second camera helps with exactly this (*Two or three cameras*).

## Custom gestures
Train your own hand gestures, e.g. to trigger avatar expressions:
```bash
.venv\Scripts\python.exe app.py gestures record peace       # 3 s countdown, then 8 s: hold it, move it around
.venv\Scripts\python.exe app.py gestures record thumbs_up
.venv\Scripts\python.exe app.py gestures record none        # relaxed hands / anything that is NOT a gesture
.venv\Scripts\python.exe app.py gestures train              # prints cross-validated accuracy + confusion matrix
```
`avatarstudio/engine/studio.py` picks the model up automatically. `gestures/actions.json` maps a gesture to an expression clip,
which is held while you show the gesture (e.g. `"peace": "Joy"`). Without a trained model, the built-in finger rules
are used (fist, open palm, peace, thumbs up/down, OK, rock on, …).

## Motion capture from a recording (offline)
For animation (Blender, Unity, Unreal) and movement analysis (sports, physio, ergonomics, exercise reps), a recording
can be processed afterwards, the way FreeMoCap does it, instead of live:
```bash
.venv\Scripts\python.exe app.py mocap take.mp4            # -> take_mocap\ next to the video
AvatarStudio.exe mocap take.mp4 --full-body           # (legs too; --cutoff 6, --max-gap 0.5, --no-wilor, --no-overlay)
```
In the launcher: **Record for mocap**, then **Mocap a video**, then **Open mocap results**.

Live tracking only knows the past: it smooths with a small delay and guesses ahead. With the whole take known:
- every frame is tracked, none dropped, and WiLoR's finger shapes are waited for on every frame (live, WiLoR skips
  frames when it falls behind);
- the avatar pose is solved without live smoothing or look-ahead, then filtered **forward and backward** in time:
  a 4th-order Butterworth low-pass at 6 Hz, FreeMoCap's default (face shapes at 12 Hz so blinks survive). There's no
  delay and no overshoot;
- body points lost for up to 0.5 s are filled with a straight line between the frames either side. Longer gaps stay
  empty rather than invented.

Output folder:
| File | What |
|---|---|
| `motion.bvh` | the avatar's motion (constant bone lengths) for Blender / MotionBuilder / Unity |
| `body_3d.csv`, `body_3d.npy` | the 33 body points per frame in metres, origin between the hips: X your right, Y forward (toward the camera), Z up; `root_*` is where you moved from your calibrated spot |
| `joint_angles.csv` | elbow, shoulder, hip and knee angles (from the body points); trunk lean, head turn / look up / tilt (against your calibrated neutral pose); wrist and finger bends (from the avatar's hands). Empty where that part wasn't tracked |
| `face_arkit.csv` | the 52 ARKit face shapes per frame (0–1), for facial animation |
| `summary.json` | share of frames each part was tracked, gaps filled, jitter before / after, every angle's range of motion and repetition count |
| `overlay.mp4` | the video with the cleaned-up skeleton drawn on it (orange = filled-in points) |

On your `take.mp4` (38 s, sitting, upper body): processed at 17.6 fps (WiLoR's ~55 ms per frame sets the pace,
tracking runs alongside it). Body-point jitter is **6.75 → 2.22 mm per frame** (median change in velocity), and 99.9%
of hand frames have WiLoR fingers. The hips were out of view, so shoulder elevation is measured against vertical, and
hip / knee angles stay empty. Nothing needed filling: the arms' gaps were at the start / end of the take (hands down,
out of view) or about 1 s long.

**Compared with FreeMoCap:** its accuracy figures (joint paths within ~3 cm, angles within 5°) come from 3–6
calibrated cameras, triangulated. One webcam judges depth from the body model's guess. Angles across the picture
(raising an arm sideways, bending an elbow facing the camera) are good. Movement toward or away from the camera is
much less certain. A 2nd and 3rd camera help (below).

### With two or three cameras
Record with the extra cameras on, and each one's video is saved next to the main one, frame for frame in step (every
time the main recording writes a frame, each extra camera's newest frame is written too):
```bash
.venv\Scripts\python.exe app.py studio --record take.mp4 --camera2 1 --camera3 2   # -> take.mp4, take_camera2.mp4, take_camera3.mp4
.venv\Scripts\python.exe app.py mocap take.mp4      # picks up take_camera2.mp4 / take_camera3.mp4 next to it by itself
```
`--video2` / `--video3` name other files, `--main-only` ignores them. In the launcher: choose the *Extra cameras*,
then **Record for mocap** and **Mocap a video** as usual.
- Every frame is tracked in every camera, side by side. WiLoR's fingers come from the main camera only.
- The cameras are lined up **from the take itself**: no checkerboard, no C needed. A rotation is fitted to the body
  points both cameras saw clearly, over the whole take (`avatarstudio/tracking/multicam.py`). If the take never shows you to both cameras,
  the alignment saved by your last C calibration is used instead.
- Then each frame is merged as live (body points averaged by how clearly each camera saw them; a hand or your face
  only a side camera saw is added) before the avatar pose is solved, filtered and exported. Axes, root motion and
  `overlay.mp4` stay the main camera's: the overlay draws what the main camera saw.
- `summary.json` adds `cameras` (for each camera: lined up or why not, `error_cm` = average fit, `angle_deg` from
  the main camera, `mirror_t`, its share of the body), `body_points_from_side_cameras` (`weight_share`: the side
  cameras' share of the merge; `seen_only_by_side_cameras`: points the main camera missed) and
  `frames_only_side_cameras_saw` (body / face / each hand).
- **Mirrored cameras are left out.** A phone's front camera or a webcam with mirroring on would put its view of your
  left arm on your right. The body model swaps left and right on a mirrored picture too, so the mirror only shows
  when you move one side differently from the other. Over a take, the mirror image and the rotation are compared
  stretch by stretch (`mirror_t` below −2: mirrored, not used). Holding still for C can't tell them apart, so turn
  mirroring off in the camera's or phone app's settings.
- Tested with synthetic takes made from take.mp4:
  - **The same video twice:** lines up at 0° / 0.0 cm, and every output is byte-identical to one camera's.
  - **A copy rolled 20° about its own lens axis:** lines up at 20.7° with a 3.4 cm fit and moves body points by
    1.1 cm (median); raw jitter goes 6.42 → 5.6 mm and left-hand frames 77% → 84%.
  - **A mirrored copy:** rejected (`mirror_t` −7.6 on the whole take, −2.5 on a 10 s clip), also as the 3rd camera
    next to a good 2nd one; the results are then byte-identical to leaving it out. The rolled copy was never
    rejected in any 3 s or 5 s stretch.
  - **The main camera blacked out for 1 s:** the side camera carried body, face and hands through it (2.3 cm from
    the two-camera result).

  Single-camera results are byte-identical to before. Speed: 17.8 fps with three cameras vs 20.5 fps with one on the
  whole take (WiLoR sets the pace). Memory: about 10 KB per frame plus 3 KB per extra camera, ~270 MB for 10 minutes
  with three cameras. **Not tested with real extra webcams yet**: a real side view differs more than these copies.
- This is merging, not FreeMoCap's triangulation. Each camera's 3D is still its body model's guess (lined up by
  rotation only, no lens or position calibration), so averaging reduces errors and fills in what the main camera
  couldn't see, but depth isn't measured.

## Tracking only (no avatar)
```bash
.venv\Scripts\python.exe -m avatarstudio.tracking.all_in_one --webcam         # q quit, m mesh, b body, r reset depth
.venv\Scripts\python.exe -m avatarstudio.tracking.all_in_one photo.jpg        # annotated photo + JSON summary
```
Per frame: 478-point face mesh, head yaw/pitch/roll, gaze, eyes open/closed, mouth, smile, 33-point body
with arm/leg angles and body turn, both hands with per-finger curl, gestures, pinch distance, and depth: distance
from the camera, moving closer/away, and each hand's distance.

## How it works
| Piece | File |
|---|---|
| MediaPipe face, hand and body landmarkers, merged per frame | `avatarstudio/tracking/all_in_one.py` |
| Optional: MediaPipe's models on the GPU, its tracking copied (+ the model converter) | `avatarstudio/tracking/gpu_tracking.py`, `tools/convert_landmark_models.py` |
| Experimental 2nd / 3rd camera: lining up and merging | `avatarstudio/tracking/multicam.py` |
| The one-file app's unpacker | `packaging/single_exe.cs` |
| Landmarks → VRM bone rotations, root motion, VRM/ARKit blendshapes, calibration | `avatarstudio/avatar/rig.py` |
| One Euro smoothing | `avatarstudio/tracking/filters.py` |
| VMC (OSC/UDP) sender, WebSocket stream, viewer web server | `avatarstudio/engine/streaming.py` |
| Live 3D viewer (three.js + three-vrm, bundled in `avatarstudio/ui/web/lib/`) and settings page | `avatarstudio/ui/web/index.html`, `avatarstudio/ui/web/settings.html` |
| The one-window app: engine control, devices, tools; its page | `avatarstudio/ui/mainwindow.py`, `avatarstudio/ui/web/app.html`, `avatarstudio/ui/web/app.js`, `avatarstudio/ui/web/app.css` |
| The camera picture for the app (`studio --ui`) | `avatarstudio/engine/preview.py` |
| Updates from GitHub releases; publishing one | `avatarstudio/core/updater.py`, `avatarstudio/version.py`, `tools/release.py` |
| The viewer and settings page as app windows (WebView2 via pywebview) | `avatarstudio/ui/webwindow.py` |
| What differs between Windows, macOS and Linux (GPU runtime per OS, opening folders) | `avatarstudio/core/os_support.py` |
| Microphone lip sync (loudness + formants) | `avatarstudio/audio/mic.py` |
| Settings, profiles, hotkeys | `avatarstudio/core/settings.py` |
| VTube Studio plugin client | `avatarstudio/engine/vts.py` |
| Threaded webcam reader with auto-reconnect | `avatarstudio/tracking/camera.py` |
| Validation test pattern | `avatarstudio/avatar/testpattern.py` |
| Clap latency test | `avatarstudio/engine/latency.py` |
| Gesture recording / training / recognition | `avatarstudio/personal/gestures.py` |
| BVH export | `avatarstudio/avatar/bvh.py` |
| Offline motion capture: zero-lag filtering, gap filling, CSV / BVH / overlay exports | `avatarstudio/mocap.py` |
| The app tying it together | `avatarstudio/engine/studio.py` |

**Depth:** distance comes from the gap between your pupils (about 6.3 cm on adults), or from your body's 3D size when
your face isn't visible. The two are blended over 0.75 s when switching between them, and the body estimate is
continuously rescaled to match the eyes. Replaying a recorded session, this cut the jump at a switch from 10.6% to
2.0%. Changes in distance are precise. The absolute distance depends on the camera's field of view and your eye
spacing (Settings → Camera and distance, 65° and 6.3 cm by default): with the defaults it can be 10–15% off. *Measure
it* there takes your real distance to the camera (measured to your eyes, looking straight at it) and sets the field
of view so the reading matches: the distance scales with the focal length, so the new field of view is
2·atan(tan(old/2) · measured / real). Changing either setting while tracking rescales the smoothed distance, its
reference and the calibration's distance by the same factor, so the avatar doesn't move.

## Checks
- `tests/test_rig.py` rebuilds the skeleton from the bone rotations it computes (forward kinematics) and compares each
  limb with the tracked one. On the sample photos every arm, hand and head bone lands within 1.4°, and
  `--mirror-avatar` is an exact left/right reflection. Fingers match exactly on open hands. On the tightly gripping
  hands they differ by up to 13°, which is the joint limits correcting impossible bends (see Hands and fingers).
- The full pipeline was tested headless on sample photos: a VMC receiver got all bones and blendshapes, the WebSocket
  viewer rendered the pose, and the BVH file read back to within 0.001 cm of the rig.
- **Not yet tested inside a VTuber app.** The bone convention follows the VMC/VRM standard (Unity space, T-pose =
  identity), but each app and model can differ. If a limb bends the wrong way in your app, that's the first thing
  to check. `--mirror-avatar` and the app's own mirror settings also interact.

## Setup
See `requirements.txt`. Keep exactly one OpenCV package, version 4.x (`opencv-contrib-python<5`). The `models/`
folder holds the MediaPipe models, the GPU whole-body model, `silero_vad.onnx` (speech detection),
`rnnoise.dll` (noise suppression, taken from the `pyrnnoise` wheel), and `backbone.onnx` (mouth model
features, made by `tools/export_backbone.py`).

## License
Avatar Studio's own code is under the [MIT licence](LICENSE). The libraries and models it uses keep their own
licences, listed in **[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)**; the app carries them in its `licenses`
folder. Everything in the app allows free sharing with credit, but the whole-body model and the mouth model were
trained on datasets meant for non-commercial research (COCO-WholeBody, ImageNet), and the optional WiLoR fingers
(WiLoR, MANO) are licensed for non-commercial use only. So Avatar Studio is a hobby project for personal,
non-commercial use: don't sell the app or a version of it, and check those licences yourself before using it for
anything that earns money, such as monetized streams with WiLoR fingers on.
