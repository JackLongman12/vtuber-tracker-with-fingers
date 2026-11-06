"""Train your own hand gestures from a few seconds of webcam recording each.

  .venv\\Scripts\\python.exe app.py gestures record peace        # hold the gesture (either hand), move it around a bit
  .venv\\Scripts\\python.exe app.py gestures record none         # IMPORTANT: relaxed / random hands that are no gesture
  .venv\\Scripts\\python.exe app.py gestures train               # trains + reports accuracy
  .venv\\Scripts\\python.exe app.py gestures list
  .venv\\Scripts\\python.exe app.py gestures delete peace

The model uses each hand's 21 3D landmarks (wrist-centred, scale-normalised; left hands are mirrored so one
model covers both). studio.py loads it automatically; gestures/actions.json maps gestures to avatar expressions.
"""
import argparse
import json
import os
import time
from collections import Counter, deque

import cv2
import joblib
import numpy as np
from avatarstudio.core.paths import user_file

DIR = user_file("gestures")
DATA = os.path.join(DIR, "data")
MODEL = os.path.join(DIR, "model.joblib")
ACTIONS = os.path.join(DIR, "actions.json")
DEFAULT_ACTIONS = {
    "_help": "gesture name -> VRM expression clip (Joy, Angry, Sorrow, Fun, Surprised, or any clip on your model) "
             "held at full strength while the gesture is shown",
    "peace": "Joy",
    "thumbs_up": "Fun",
}


def features(hand_world, side):
    """(21, 3) hand world landmarks -> 63-d vector that doesn't depend on hand size or which hand."""
    p = np.asarray(hand_world, dtype=np.float32) - hand_world[0]
    if side == "left":
        p[:, 0] *= -1
    return (p / (np.linalg.norm(p[9]) + 1e-6)).ravel()


def augment(x, n=10, rng=np.random.default_rng(0)):
    """Extra training samples: small 3D rotations, scale and landmark jitter."""
    out = [x]
    pts = x.reshape(-1, 21, 3)
    for _ in range(n):
        a = rng.normal(0, 0.15, 3)
        cx, cy, cz, sx, sy, sz = *np.cos(a), *np.sin(a)
        r = (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
             @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))
        q = pts @ r.T * rng.uniform(0.95, 1.05) + rng.normal(0, 0.02, pts.shape)
        out.append(q.reshape(len(x), -1))
    return np.concatenate(out)


def record(name, seconds, camera, ui=False, video=None):
    """ui: shown in the Avatar Studio app (preview.ToolView) instead of a window of its own. video: a recording
    instead of the camera, played in a loop (for checks)."""
    from avatarstudio.tracking.all_in_one import AllInOneTracker, draw
    os.makedirs(DATA, exist_ok=True)
    tracker = AllInOneTracker(video=True, max_faces=1)
    if ui:
        from avatarstudio.engine.preview import ToolView
        ui = ToolView("gesture recording")
    cap = cv2.VideoCapture(video or camera)
    samples, start = [], time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            if video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            raise SystemExit("Could not read from the camera")
        res = tracker(frame)
        elapsed = time.time() - start
        recording = 3 <= elapsed < 3 + seconds
        if recording:
            samples += [features(h["_world"], h["side"]) for h in res["hands"]]
        view = draw(frame, res, mirror=True, mesh=False, body=False, hud=False)
        if ui:
            what = "relaxed hands that aren't a gesture" if name == "none" else f"your \"{name}\" gesture"
            prompt = ([f"Get ready to show {what}", f"Recording starts in {3 - elapsed:.0f} s"] if elapsed < 3 else
                      [f"Show {what}, moving your hand around a bit",
                       f"{max(0, 3 + seconds - elapsed):.0f} s left  ·  {len(samples)} samples"])
            ui.show(view, prompt, progress=min(max(elapsed - 3, 0) / seconds, 1.0), hands=len(res["hands"]))
            stop = "q" in ui.keys()
        else:
            msg = (f"get ready: {name} in {3 - elapsed:.0f}" if elapsed < 3 else
                   f"RECORDING {name}: {3 + seconds - elapsed:.0f}s  ({len(samples)} samples)" if recording else "done")
            cv2.putText(view, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if recording else (0, 255, 255), 2)
            cv2.imshow("Record gesture", view)
            stop = cv2.waitKey(1) & 0xFF == ord("q")
        if stop or elapsed >= 3 + seconds:
            break
    cap.release()
    cv2.destroyAllWindows()
    if not samples:
        raise SystemExit("No hands were seen, nothing saved.")
    path = os.path.join(DATA, f"{name}.npy")
    old = np.load(path) if os.path.exists(path) else np.zeros((0, 63), np.float32)
    np.save(path, np.concatenate([old, np.array(samples, np.float32)]))
    print(f"saved {len(samples)} samples for '{name}' ({len(old) + len(samples)} total) -> {path}")


def load_data():
    names = sorted(f[:-4] for f in os.listdir(DATA) if f.endswith(".npy")) if os.path.isdir(DATA) else []
    xs = [np.load(os.path.join(DATA, f"{n}.npy")) for n in names]
    return names, xs


def train():
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix

    names, xs = load_data()
    if len(names) < 2:
        raise SystemExit("Record at least two gestures first (one of them should be 'none').")
    if "none" not in names:
        print("warning: no 'none' gesture recorded, so every hand will be classified as one of your gestures")
    y = np.concatenate([[i] * len(x) for i, x in enumerate(xs)])
    x = np.concatenate(xs)
    make = lambda: make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=600, early_stopping=True,
                                                                 random_state=0))
    # honest accuracy: cross-validation on contiguous chunks, so near-identical neighbouring frames
    # don't end up on both sides of the split
    groups = np.zeros(len(y), int)
    for i in range(len(names)):
        idx = np.where(y == i)[0]
        groups[idx] = np.arange(len(idx)) * 5 // max(len(idx), 1)
    pred = np.zeros_like(y)
    for fold in range(5):
        test = groups == fold
        m = make().fit(augment(x[~test]), np.tile(y[~test], 11))
        pred[test] = m.predict(x[test])
    acc = (pred == y).mean()
    print(f"cross-validated accuracy: {acc:.1%} over {len(y)} samples, {len(names)} gestures")
    cm = confusion_matrix(y, pred)
    w = max(len(n) for n in names)
    print(" " * (w + 2) + " ".join(f"{n[:6]:>6}" for n in names))
    for n, row in zip(names, cm):
        print(f"{n:>{w}}: " + " ".join(f"{v:6d}" for v in row))
    model = make().fit(augment(x), np.tile(y, 11))
    joblib.dump({"model": model, "names": names}, MODEL)
    if not os.path.exists(ACTIONS):
        with open(ACTIONS, "w") as f:
            json.dump(DEFAULT_ACTIONS, f, indent=2)
    print(f"saved -> {MODEL}")


class GestureRecognizer:
    """Per-hand prediction with a short majority vote so the label doesn't flicker."""

    def __init__(self, threshold=0.7, window=7):
        self.ok = os.path.exists(MODEL)
        if self.ok:
            try:
                d = joblib.load(MODEL)
                self.model, self.names = d["model"], d["names"]
            except Exception as e:  # a damaged / incompatible model file: built-in gestures instead of a crash
                print(f"gesture model unreadable ({e}); using the built-in gestures - retrain with gestures.py train")
                self.ok = False
        from avatarstudio.core.paths import load_json
        actions = load_json(ACTIONS, {}) if os.path.exists(ACTIONS) else {}
        self.actions = {k: v for k, v in actions.items() if not k.startswith("_")} if isinstance(actions, dict) else {}
        self.threshold = threshold
        self.history = {"left": deque(maxlen=window), "right": deque(maxlen=window)}

    def __call__(self, hands):
        """Returns {side: gesture or ''} and the expression clips the active gestures trigger."""
        out = {}
        if not self.ok:
            return out, {}
        seen = set()
        for h in hands:
            p = self.model.predict_proba(features(h["_world"], h["side"])[None])[0]
            k = int(p.argmax())
            label = self.names[k] if p[k] >= self.threshold and self.names[k] != "none" else ""
            hist = self.history[h["side"]]
            hist.append(label)
            best, count = Counter(hist).most_common(1)[0]
            out[h["side"]] = best if count > len(hist) // 2 else ""
            seen.add(h["side"])
        for side in self.history:
            if side not in seen:
                self.history[side].clear()
        shapes = {self.actions[g]: 1.0 for g in out.values() if g in self.actions}
        return out, shapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["record", "train", "list", "delete"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--ui", action="store_true", help="record in the Avatar Studio app instead of a window of its own")
    ap.add_argument("--input", help="a recording instead of the camera, played in a loop (for checks)")
    args = ap.parse_args()
    if args.command == "record":
        if not args.name:
            raise SystemExit("usage: gestures.py record NAME")
        record(args.name, args.seconds, args.camera, ui=args.ui, video=args.input)
    elif args.command == "train":
        train()
    elif args.command == "list":
        names, xs = load_data()
        for n, x in zip(names, xs):
            print(f"{n}: {len(x)} samples")
        print(f"model: {'trained' if os.path.exists(MODEL) else 'not trained yet'}")
    elif args.command == "delete":
        os.remove(os.path.join(DATA, f"{args.name}.npy"))
        print(f"deleted '{args.name}' (run train again)")


if __name__ == "__main__":
    main()
