"""Personal face refiner: a ~90 s guided recording teaches the tracker how YOUR face reads in MediaPipe.

  .venv\\Scripts\\python.exe app.py train-face          (or double-click "Train Face.bat")

For every face shape it learns:
  - your resting level (from the neutral face), which is subtracted
  - your own strongest version (from the prompt that asks for it), which becomes 1.0
  - cross-talk: shapes MediaPipe wrongly raises when you make *other* expressions (e.g. eyes half-closing in a
    big smile), learned with non-negative least squares and subtracted
It checks itself on half the recording it didn't train on and only saves if that is no worse. The studio loads
face_refiner.json automatically (switch "Personal face refiner" on the settings page).
"""
import json
import os
import time

import numpy as np
from avatarstudio.core.paths import user_file

REFINER_PATH = user_file("face_refiner.json")
# (instruction, ARKit shapes this expression should drive)
PROMPTS = [
    ("Relax your face (neutral)", []),
    ("Smile with your mouth closed", ["mouthSmileLeft", "mouthSmileRight"]),
    ("Big smile, show your teeth", ["mouthSmileLeft", "mouthSmileRight", "cheekSquintLeft", "cheekSquintRight"]),
    ("Sad mouth: corners down", ["mouthFrownLeft", "mouthFrownRight"]),
    ("Raise your eyebrows high", ["browInnerUp", "browOuterUpLeft", "browOuterUpRight"]),
    ("Frown hard: eyebrows down", ["browDownLeft", "browDownRight"]),
    ("Eyes wide open", ["eyeWideLeft", "eyeWideRight"]),
    ("Squint both eyes", ["eyeSquintLeft", "eyeSquintRight"]),
    ("Close both eyes gently", ["eyeBlinkLeft", "eyeBlinkRight"]),
    ("Close only your LEFT eye", ["eyeBlinkLeft"]),
    ("Close only your RIGHT eye", ["eyeBlinkRight"]),
    ("Puff your cheeks", ["cheekPuff"]),
    ("Pucker your lips (kiss)", ["mouthPucker"]),
    ("Round your lips like saying 'oh'", ["mouthFunnel"]),
    ("Open your jaw wide", ["jawOpen"]),
    ("Push your mouth to YOUR left", ["mouthLeft"]),
    ("Push your mouth to YOUR right", ["mouthRight"]),
    ("Press your lips together hard", ["mouthPressLeft", "mouthPressRight"]),
    ("Wrinkle your nose (sneer)", ["noseSneerLeft", "noseSneerRight"]),
    ("Relax your face again (neutral)", []),
]
READY_S, HOLD_S = 1.5, 2.5
# extra prompts for the mouth model (mouth_model.py), recorded after the ones above: (instruction, shape, seconds)
MOUTH_PROMPTS = [
    ("Stick your tongue out", "tongueOut", 3.0),
    ("Talk normally: count from one to ten", None, 5.0),  # speech must not look like tongue / puff / funnel
    ("Tongue out again, a bit to one side", "tongueOut", 2.5),
]
# which mouth class each refiner prompt teaches (None = skip: a kiss looks too much like "oh")
MOUTH_LABEL = {"Puff your cheeks": "cheekPuff", "Round your lips like saying 'oh'": "mouthFunnel",
               "Pucker your lips (kiss)": None}


def fit(frames):
    """frames: list of (prompt_index, {shape: value}) from the hold phase of each prompt -> refiner params."""
    from scipy.optimize import nnls
    names = sorted(frames[0][1])
    X = np.array([[f[1].get(n, 0.0) for n in names] for f in frames])
    P = np.array([f[0] for f in frames])
    neutral = np.isin(P, [i for i, (_, t) in enumerate(PROMPTS) if not t])
    base = np.median(X[neutral], axis=0) if neutral.any() else np.zeros(len(names))
    params = {}
    for j, s in enumerate(names):
        tgt = np.isin(P, [i for i, (_, t) in enumerate(PROMPTS) if s in t])
        if not tgt.any():
            params[s] = {"base": float(base[j]), "peak": 1.0, "cross": {}}
            continue
        peak = float(np.percentile(X[tgt, j], 90))
        if peak - base[j] < 0.05:  # MediaPipe barely moves this shape for you: leave it alone
            params[s] = {"base": float(base[j]), "peak": float(base[j] + 1.0), "cross": {}}
            continue
        # cross-talk: when you do OTHER expressions, how much of this shape is explained by their shapes
        other = ~tgt & ~neutral
        drivers = sorted({k for i in set(P[other]) for k in PROMPTS[i][1] if k != s and k in names})
        cross = {}
        if drivers and other.sum() > 10:
            A = np.clip(X[other][:, [names.index(k) for k in drivers]] - base[[names.index(k) for k in drivers]], 0, None)
            r = np.clip(X[other, j] - base[j], 0, None)
            coef, _ = nnls(A, r)
            # never let the correction eat into the real expression: cap so it removes <= 15% of the peak there
            At = np.clip(X[tgt][:, [names.index(k) for k in drivers]] - base[[names.index(k) for k in drivers]], 0, None)
            removed = np.median(At @ coef) if len(At) else 0.0
            if removed > 0.15 * (peak - base[j]):
                coef *= 0.15 * (peak - base[j]) / removed
            cross = {k: float(c) for k, c in zip(drivers, coef) if c > 1e-3}
        params[s] = {"base": float(base[j]), "peak": peak, "cross": cross}
    return params


def apply(params, shapes):
    """Refine a {shape: value} dict with fitted params (unknown shapes pass through)."""
    out = dict(shapes)
    for s, p in params.items():
        if s not in shapes:
            continue
        v = shapes[s] - sum(c * max(shapes.get(k, 0.0) - params[k]["base"], 0.0) for k, c in p["cross"].items())
        out[s] = float(np.clip((v - p["base"]) / max(p["peak"] - p["base"], 1e-3), 0, 1))
    return out


def evaluate(params, frames):
    """How well target expressions reach full strength, and how much cross-talk remains, before vs after."""
    tgt_before, tgt_after, ct_before, ct_after = [], [], [], []
    for i, shapes in frames:
        want = PROMPTS[i][1]
        if not want:
            continue
        ref = apply(params, shapes)
        for s in shapes:
            if s.startswith("_"):
                continue
            if s in want:
                tgt_before.append(shapes[s])
                tgt_after.append(ref[s])
            elif s in params and params[s]["peak"] - params[s]["base"] < 1.0:  # a learned shape, not asked for
                ct_before.append(shapes[s])
                ct_after.append(ref[s])
    m = lambda a: float(np.median(a)) if a else float("nan")
    return {"target_before": m(tgt_before), "target_after": m(tgt_after),
            "crosstalk_before": float(np.mean(ct_before)) if ct_before else float("nan"),
            "crosstalk_after": float(np.mean(ct_after)) if ct_after else float("nan")}


def train_and_save(frames, path=REFINER_PATH):
    """Fit on even frames of each prompt, check on odd ones, then refit on everything and save."""
    train = [f for k, f in enumerate(frames) if k % 2 == 0]
    test = [f for k, f in enumerate(frames) if k % 2 == 1]
    report = evaluate(fit(train), test)
    params = fit(frames)
    improved = report["target_after"] >= report["target_before"] and report["crosstalk_after"] <= report["crosstalk_before"]
    if improved:
        from avatarstudio.core.paths import save_json
        save_json(path, {"created": time.strftime("%Y-%m-%d %H:%M"), "params": params, "check": report}, indent=1)
    return params, report, improved


def load(path=REFINER_PATH):
    """The saved refiner, or None (none trained, or the file is damaged - then it's set aside, not fatal)."""
    from avatarstudio.core.paths import load_json
    data = load_json(path, None)
    return data.get("params") if isinstance(data, dict) else None


def mouth_label(i):
    """Mouth model class of step i of the full recording (refiner prompts, then MOUTH_PROMPTS); -1 = skip."""
    from avatarstudio.personal.mouth_model import CLASSES
    if i < len(PROMPTS):
        lab = MOUTH_LABEL.get(PROMPTS[i][0], "none")
    else:
        lab = MOUTH_PROMPTS[i - len(PROMPTS)][1] or "none"
    return CLASSES.index(lab) if lab else -1


def record(camera=0, ui=False, video=None):
    """Returns (refiner frames [(prompt, shapes)], mouth samples (crops, labels, groups)).
    ui: shown in the Avatar Studio app (preview.ToolView) instead of a window of its own. video: a recording
    instead of the camera, played in a loop (for checks)."""
    import cv2
    from avatarstudio.tracking.all_in_one import AllInOneTracker, draw
    from avatarstudio.tracking.camera import CameraStream
    from avatarstudio.personal.mouth_model import mouth_crop
    tracker = AllInOneTracker(video=True, max_faces=1)
    if ui:
        from avatarstudio.engine.preview import ToolView
        ui = ToolView("face training")
    cam, cap = (None, cv2.VideoCapture(video)) if video else (CameraStream(camera), None)
    steps = [(text, HOLD_S) for text, _ in PROMPTS] + [(text, secs) for text, _, secs in MOUTH_PROMPTS]
    frames, crops, labels, groups = [], [], [], []
    i, t0, last, n = 0, time.time(), 0, 0
    while i < len(steps):
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        else:
            ok, frame, last = cam.read(last)
            if not ok:
                continue
        frame = frame.copy()
        res = tracker(frame)
        dt = time.time() - t0
        text, hold = steps[i]
        holding = dt >= READY_S
        face = res["faces"][0] if res["faces"] else None
        if holding and face and face.get("blendshapes"):
            if i < len(PROMPTS):
                frames.append((i, dict(face["blendshapes"])))
            n += 1
            if n % 2 == 0 and mouth_label(i) >= 0:
                crops.append(mouth_crop(frame, face))
                labels.append(mouth_label(i))
                groups.append(i)
        view = draw(frame, res, mirror=True, body=False, hud=False)
        if ui:  # (the app shows the prompt and its progress over the picture)
            ui.show(view, [text, f"{i + 1} of {len(steps)}  ·  {'hold it' if holding else 'get ready'}"]
                    + ([] if face else ["Look at the camera: your face isn't visible"]),
                    progress=min(dt / (READY_S + hold), 1.0), face=bool(face))
            cancel = "q" in ui.keys()
        else:
            h, w = view.shape[:2]
            cv2.rectangle(view, (0, 0), (w, 90), (20, 20, 20), -1)
            cv2.putText(view, f"{i + 1}/{len(steps)}  {text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (80, 230, 255) if holding else (180, 180, 180), 2, cv2.LINE_AA)
            bar = int(w * min(dt / (READY_S + hold), 1))
            cv2.rectangle(view, (0, 80), (bar, 90), (80, 230, 255) if holding else (120, 120, 120), -1)
            if not face:
                cv2.putText(view, "face not found - look at the camera", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)
            cv2.imshow("Face refiner - follow the prompts (Q to cancel)", view)
            cancel = cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"))
        if cancel:
            (cam or cap).release()
            cv2.destroyAllWindows()
            raise SystemExit("cancelled")
        if dt >= READY_S + hold:
            i, t0 = i + 1, time.time()
    (cam or cap).release()
    cv2.destroyAllWindows()
    return frames, (crops, labels, groups)


def train_mouth(crops, labels, groups):
    from avatarstudio.personal import mouth_model
    np.savez_compressed(mouth_model.DATA_PATH, crops=np.asarray(crops, np.uint8), labels=np.asarray(labels),
                        groups=np.asarray(groups))
    print("training the tongue / cheek puff / funnel model...")
    report = mouth_model.train_and_save(crops, labels, groups)
    if report is None:
        print("  not enough mouth pictures (was your face visible?)")
        return
    for s in mouth_model.SHAPES:
        found, right = report["per_class"].get(s, 0), report["precision"].get(s, 0)
        state = "ON" if s in report["enabled"] else "off (not reliable enough - record again)"
        print(f"  {s}: found {found * 100:.0f}% of the time, {right * 100:.0f}% of detections right -> {state}")
    if report["enabled"]:
        print(f"saved -> {mouth_model.MODEL_PATH}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--ui", action="store_true", help="shown in the Avatar Studio app instead of a window of its own")
    ap.add_argument("--input", help="a recording instead of the camera, played in a loop (for checks)")
    args = ap.parse_args()
    frames, mouth = record(args.camera, ui=args.ui, video=args.input)
    seen = {i for i, _ in frames}
    missing = [PROMPTS[i][0] for i in range(len(PROMPTS)) if i not in seen]
    if missing:
        print(f"face wasn't visible during: {missing}")
    if len(frames) < 200:
        raise SystemExit("too little face data recorded; try again facing the camera in good light")
    _, report, saved = train_and_save(frames)
    print(f"check on unseen frames: asked-for expressions reach {report['target_before']:.2f} -> "
          f"{report['target_after']:.2f} (1.0 = full), cross-talk {report['crosstalk_before']:.3f} -> "
          f"{report['crosstalk_after']:.3f}")
    print(f"saved -> {REFINER_PATH}" if saved else "not saved: it did not improve on the unseen frames")
    train_mouth(*mouth)


if __name__ == "__main__":
    main()
