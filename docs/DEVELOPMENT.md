# Developing Avatar Studio

How the project is laid out, how to run it from source, and how to build and publish a new version. What the app
does and how well is in the [README](../README.md).

## Layout

```
app.py                     the entry point: `python app.py` (the app window) or `python app.py <command>`
avatarstudio/              the application (one Python package)
  main.py                  the commands: studio, mocap, train-face, gestures, tune-smoothing, viewer, settings, ...
  version.py               VERSION, and the GitHub repository updates come from
  mocap.py                 offline motion capture of a video (BVH, CSV, face curves, overlay video)
  core/                    used everywhere
    paths.py               where things are: your data, models/, the web pages
    settings.py            the live settings, profiles, hotkeys
    os_support.py          what differs between Windows, macOS and Linux (GPU runtime, opening folders)
    perf.py                keeps the OS from slowing the app down in the background
    updater.py             updates from GitHub releases
  tracking/                camera in, landmarks out
    all_in_one.py          MediaPipe face + hands + body per frame, merged (the tracker)
    gpu_tracking.py        MediaPipe's models on the GPU (ONNX Runtime)
    wholebody.py           the GPU whole-body model (left / right of crossing hands)
    multicam.py            2nd / 3rd camera: lining up and merging
    camera.py              threaded webcam reader with reconnects
    filters.py             One Euro smoothing
    crop_model.py          small image models on crops (shared by the mouth model)
    wilor_client.py        WiLoR finger shapes, from a helper process...
    wilor_server.py        ...that runs in .venv_wilor (PyTorch), outside this package
  avatar/                  landmarks in, avatar out
    rig.py                 bone rotations, root motion, VRM / ARKit blendshapes, calibration
    bvh.py                 BVH export
    testpattern.py         the avatar test's 24 known poses
  audio/
    mic.py                 microphone lip sync (loudness, formants, voice calibration)
    voice.py               speech detection (Silero VAD) and noise suppression (RNNoise)
  personal/                learning the user
    face_refiner.py        face training: how your face reads, tongue / cheek model recording
    mouth_model.py         tongue out / cheek puff / funnel classifier
    gestures.py            custom hand gestures: record, train, recognise
    tune_smoothing.py      smoothing tuned to a recorded take
  engine/                  the live tracking process
    studio.py              camera -> tracking -> rig -> outputs, calibration, keys (the engine)
    streaming.py           outputs: VMC (OSC/UDP), WebSocket JSON, the web server on :8000
    preview.py             the camera picture as a JPEG stream for the app window (studio --ui, tools --ui)
    latency.py             the clap latency test
    vts.py                 VTube Studio plugin client
  ui/                      what the user sees
    mainwindow.py          the app window: starts the engine and tools, devices, settings, updates (Python side)
    webwindow.py           web pages as native windows (pywebview), the local web server
    launcher.py            the classic launcher (Tk), used where no web view is available
    web/                   the pages: app.html/.js/.css (the app), index.html (3D viewer, also for OBS),
                           settings.html, lib/ (three.js, three-vrm: bundled so nothing loads from the internet)
tools/                     for developers, not shipped
  build_dist.py            builds Distributions/AvatarStudio.exe (one file)
  release.py               publishes a version on GitHub (the installed apps update to it)
  fetch_models.py          downloads models/ after cloning
  convert_landmark_models.py, export_backbone.py   make the GPU / mouth model files in models/
  compare_tracking.py, wilor_bench.py, wilor_fix_mano.py   measuring and WiLoR setup
tests/test_rig.py          rig self-test on the photos in tests/images (forward kinematics, mirroring)
packaging/                 the one-file launcher (single_exe.cs), the icon, licenses/ (the models' licence texts)
scripts/windows/           double-click shortcuts for development runs (.bat)
models/                    model files (not in git: too big; tools/fetch_models.py)
data/                      your own data in development runs (not in git)
LICENSE                    MIT, for the project's own code
THIRD_PARTY_NOTICES.md     every library and model the app uses, with its licence
```

How the pieces talk: the app window (`ui/mainwindow.py` + `ui/web/app.*`) starts the engine (`app.py studio --ui`) as
a separate process. The engine sends avatar data and status over a WebSocket (ws://127.0.0.1:8765), its camera
picture as camera.mjpg on http://127.0.0.1:8000, and takes settings, keys and button presses on the same
WebSocket. Face training and gesture recording run the same way while tracking is off (`--ui`). VTuber apps get
the avatar over VMC (UDP 39539); OBS shows http://127.0.0.1:8000.

## Setup

Python 3.11, Windows (macOS / Linux run from source; see the README).
```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python.exe tools/fetch_models.py
.venv\Scripts\python.exe app.py
```
`fetch_models.py` downloads `models.zip` from the repository's "models" release (whoever has the model files
publishes them with `tools/release.py --models`). WiLoR fingers are optional and need their own environment,
`.venv_wilor` with PyTorch (README "WiLoR fingers").

Where things are at run time (`avatarstudio/core/paths.py`): the working folder is the project folder (models are
found as `models/...`); your data is in `data/` (packaged: `%APPDATA%\AvatarStudio`); `AVATARSTUDIO_DATA` points
both elsewhere, which tests use so they never touch real data.

## Checks

- `tests/test_rig.py` - the rig against tracked photos; its numbers must not change unless the rig is meant to.
- Motion capture is deterministic with the GPU whole-body model and WiLoR off: `app.py mocap clip.mp4 --no-wilor`
  twice gives byte-identical files (all but `summary.json`, which has timings); a refactor must keep them so.
- The engine without a camera: `app.py studio --input take.mp4 --headless` (or `--ui` and open the app).

## Building

```bash
.venv\Scripts\python.exe tools/build_dist.py
```
makes `Distributions/AvatarStudio.exe` (Windows, about 485 MB): a PyInstaller folder build zipped onto the small
launcher in `packaging/single_exe.cs`, which unpacks a version once into `%LOCALAPPDATA%\AvatarStudio\app\` and runs
it from there. The build checks itself (models, web pages, the window's DLLs, licences) before it replaces the
previous one. It puts `LICENSE.txt`, `THIRD_PARTY_NOTICES.md` and `licenses/` next to the app: the licence files of
every package PyInstaller included (read from the packages' metadata; `licenses/INDEX.txt` lists them), Python's,
and `packaging/licenses` (the models').

## Publishing an update

Installed apps check the latest GitHub release of `GITHUB_REPO` (`avatarstudio/version.py`) when they open and
offer it when its version is newer than theirs (`avatarstudio/core/updater.py`). To publish one:

1. Once: set `GITHUB_REPO = "owner/repository"` in `avatarstudio/version.py`, and get a token: a fine-grained
   personal access token with *Contents: read and write* on the repository in `GITHUB_TOKEN`, or log in with the
   GitHub CLI (`gh auth login`).
2. Push your changes to GitHub.
3. Run
   ```bash
   .venv\Scripts\python.exe tools/release.py 1.2.0 --notes "What's new (shown in the app's update dialog)"
   ```
   It sets `VERSION` to 1.2.0, builds the app, writes `SHA256SUMS.txt`, creates the release `v1.2.0` and uploads
   `AvatarStudio.exe` and `SHA256SUMS.txt`. `--dry-run` builds without uploading.
4. Commit and push the changed `avatarstudio/version.py`.

The apps download the new `AvatarStudio.exe`, check it against `SHA256SUMS.txt`, rename the running file to
`AvatarStudio.old.exe` (Windows allows renaming a running program), put the new one in its place and start it;
the old file is removed on the next start. Pushing code alone doesn't update anyone: only a release does.
Releases with no `SHA256SUMS.txt`, or a file that doesn't match it, are not installed. The checksum guards against
damaged downloads; it doesn't prove who made the file (that would need code signing).

## Conventions

- Paths come from `avatarstudio/core/paths.py` (`user_file`, `app_dir`, `web_dir`), never from the current folder
  of a module.
- The engine and the tools stay separate processes from the app window: a crash in tracking can't take the window
  with it, and the window stays responsive.
- New settings: a field in `core/settings.py` (default + comment), a row in the `SETTINGS` list in
  `ui/web/app.js`, and the engine reads it from `settings` (changes arrive through `change_settings`).
- Measure before and after a change that could affect speed or quality (the README records the numbers).
- A new library or model: add it to `THIRD_PARTY_NOTICES.md`. A model's licence text goes in `packaging/licenses`
  (packages bring their own). Nothing licensed non-commercial or "no derivatives" (WiLoR, MANO) goes into the app,
  `models/` or the repository: users install those themselves.
