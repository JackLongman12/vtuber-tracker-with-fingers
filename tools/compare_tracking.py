"""Compare tracking setups on the same recorded video, so every upgrade has to prove itself in numbers.

Record:   .venv\\Scripts\\python.exe app.py studio --record take.mp4      (move your hands around, cross them, make fists)
Compare:  .venv\\Scripts\\python.exe tools/compare_tracking.py take.mp4

Setups: MediaPipe only / + GPU whole-body model for left/right (run while left/right is in doubt).
Metrics (lower is better unless noted):
  ms/frame      tracking time per frame
  hands %       frames with at least one hand found (higher = better)
  side flips    a hand that stayed in place but changed its left/right label, per minute
  finger hinge  how far finger middle joints bend sideways (degrees; real fingers ~0-20)
  jitter        frame-to-frame shake of fingertips relative to the wrist (mm, second difference)
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # (the project folder)
from avatarstudio.tracking.all_in_one import AllInOneTracker  # noqa: E402

SETUPS = {"MediaPipe only": dict(wholebody=False),
          "+ GPU model: left/right": dict(wholebody=True)}


def hinge_error(w):
    lat = w[5] - w[17]
    lat = lat / (np.linalg.norm(lat) + 1e-9)
    out = []
    for a, b, c in ((5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)):
        v1, v2 = w[b] - w[a], w[c] - w[b]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        if math.degrees(math.acos(np.clip(cos, -1, 1))) < 15:
            continue
        n = np.cross(v1, v2)
        n /= np.linalg.norm(n) + 1e-9
        out.append(math.degrees(math.acos(min(abs(np.dot(n, lat)), 1))))
    return out


def run(path, setup, max_frames):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tr = AllInOneTracker(video=True, max_faces=1, wholebody=setup["wholebody"])
    times, hand_frames, n, flips, hinges, jit = [], 0, 0, 0, [], []
    prev, hist = [], {}
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        res = tr(frame, timestamp_ms=n * 1000 / fps)
        times.append((time.perf_counter() - t0) * 1000)
        n += 1
        hands = res["hands"]
        hand_frames += bool(hands)
        for h in hands:
            size = np.linalg.norm(h["_pts"][0, :2] - h["_pts"][9, :2])
            for p in prev:  # same place last frame, different label = a flip
                if np.linalg.norm(p["_pts"][0, :2] - h["_pts"][0, :2]) < 0.3 * size and p["side"] != h["side"]:
                    flips += 1
            hinges += hinge_error(h["_world"])
            tips = h["_world"][[4, 8, 12, 16, 20]] - h["_world"][0]
            hs = hist.setdefault(h["side"], [])
            hs.append(tips)
            if len(hs) >= 3:
                jit.append(np.linalg.norm(hs[-1] - 2 * hs[-2] + hs[-3], axis=1).mean() * 1000)
            del hs[:-3]
        for side in list(hist):
            if side not in {h["side"] for h in hands}:
                hist.pop(side)
        prev = hands
    cap.release()
    minutes = n / fps / 60
    return {"frames": n, "ms/frame": np.median(times[5:]) if len(times) > 5 else float("nan"),
            "hands %": 100 * hand_frames / max(n, 1), "side flips/min": flips / max(minutes, 1e-9),
            "finger hinge (deg)": np.median(hinges) if hinges else float("nan"),
            "jitter (mm)": np.median(jit) if jit else float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--max-frames", type=int, default=100000)
    args = ap.parse_args()
    rows = {name: run(args.video, s, args.max_frames) for name, s in SETUPS.items()}
    keys = list(next(iter(rows.values())))[1:]
    print(f"{'setup':30s}" + "".join(f"{k:>20s}" for k in keys))
    for name, r in rows.items():
        print(f"{name:30s}" + "".join(f"{r[k]:20.1f}" for k in keys))
    print(f"({next(iter(rows.values()))['frames']} frames)")


if __name__ == "__main__":
    main()
