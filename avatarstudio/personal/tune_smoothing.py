"""Tune the motion smoothing to your own tracking noise from a recording.

  .venv\\Scripts\\python.exe app.py tune-smoothing take.mp4       (record one with: studio.py --record take.mp4)

The live filter (One Euro) can only look at the past, so it trades jitter against lag. Offline we can also look
ahead: smoothing the recording forwards and backwards (zero-phase) gives a jitter-free reference *without lag*.
For each bone group this tries many live-filter settings and keeps the one closest to that reference, i.e. the
best balance of jitter and lag for your camera, light and movements. Saved to smoothing.json (the studio loads
it automatically; delete the file to go back to the defaults) only if clearly better than the defaults.
"""
import argparse
import json
import math
import time

import cv2
import numpy as np

from avatarstudio.tracking.all_in_one import AllInOneTracker, focal_px
from avatarstudio.tracking.filters import OneEuro
from avatarstudio.avatar.rig import SMOOTH_DEFAULTS as DEFAULTS, SMOOTHING_PATH as PATH, AvatarRig, smooth_group as group_of

REF_SIGMA = 1.5  # frames: the zero-phase reference smoothing (MediaPipe's jitter is mostly frame-to-frame)


def raw_motion(video):
    """Unsmoothed bone rotations for every frame: {bone: (T, 4)}, plus timestamps."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = AllInOneTracker(video=True, max_faces=1)
    rig = AvatarRig(smoothing=False, idle=False, auto_range=False)
    seq, ts, i = {}, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        res = tracker(frame, int(t * 1000) + 1)
        out = rig(res, focal_px(frame.shape[1], tracker.fov), t)
        for b, q in out["bones"].items():
            seq.setdefault(b, {})[i] = np.asarray(q, float)
        ts.append(t)
        i += 1
    n = len(ts)
    bones = {}
    for b, d in seq.items():
        if len(d) < n * 0.5:
            continue
        cur, rows = next(iter(d.values())), []
        for k in range(n):  # frames without this bone repeat the last value
            cur = d.get(k, cur)
            rows.append(cur)
        q = np.array(rows)
        for k in range(1, n):  # one hemisphere, like the live filter
            if np.dot(q[k], q[k - 1]) < 0:
                q[k] = -q[k]
        bones[b] = q
    return bones, np.array(ts)


def zero_phase(q, sigma=REF_SIGMA):
    k = np.arange(-int(4 * sigma), int(4 * sigma) + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= w.sum()
    pad = np.pad(q, ((len(k) // 2, len(k) // 2), (0, 0)), mode="edge")
    out = np.stack([np.convolve(pad[:, c], w, "valid") for c in range(q.shape[1])], 1)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def angle_deg(a, b):
    return np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1)))


def run_filter(q, ts, min_cutoff, beta, lead=0.0):
    f = OneEuro(min_cutoff=min_cutoff, beta=beta, lead=lead)
    out = np.empty_like(q)
    for k in range(len(q)):
        out[k] = f(q[k], ts[k])
    return out


STILL_DEG_S = 15.0  # a bone turning slower than this (in the reference) counts as held still


def score(bones, refs, ts, min_cutoff, beta):
    """(error, shake) for one setting over a group's bones:
    error = mean angle (deg) between the live output and the lag-free reference (lag + leftover jitter);
    shake = mean frame-to-frame turn (deg) of the live output while the bone is held still - the visible
    trembling. Frames where the bone wasn't tracked (value held) are left out."""
    errs, shakes = [], []
    for b, q in bones.items():
        live = run_filter(q, ts, min_cutoff, beta)
        live = live / np.linalg.norm(live, axis=1, keepdims=True)
        tracked = np.r_[True, np.any(np.abs(np.diff(q, axis=0)) > 1e-9, axis=1)]
        errs.append(angle_deg(live, refs[b])[tracked])
        ref_speed = np.r_[0, angle_deg(refs[b][1:], refs[b][:-1]) / np.diff(ts)]
        still = (ref_speed < STILL_DEG_S) & tracked
        still[0] = False
        shakes.append(np.r_[0, angle_deg(live[1:], live[:-1])][still])
    return float(np.mean(np.concatenate(errs))), float(np.mean(np.concatenate(shakes)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--min-gain", type=float, default=0.05, help="save only if at least this much better")
    args = ap.parse_args()
    t0 = time.time()
    bones, ts = raw_motion(args.video)
    print(f"tracked {len(ts)} frames in {time.time() - t0:.0f} s")
    refs = {b: zero_phase(q) for b, q in bones.items()}
    cutoffs = [0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0, 7.0, 10.0]
    betas = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    tuned, report = {}, {}
    for g, (mc0, b0) in DEFAULTS.items():
        gb = {b: q for b, q in bones.items() if group_of(b) == g}
        if not gb:
            continue
        rg = {b: refs[b] for b in gb}
        base_err, base_shake = score(gb, rg, ts, mc0, b0)
        raw_err, raw_shake = score(gb, rg, ts, 1e6, 0.0)  # no smoothing at all
        # least lag/error among settings that shake no more than the current ones when you hold still
        cands = [(*score(gb, rg, ts, mc, be), mc, be) for mc in cutoffs for be in betas]
        ok = [c for c in cands if c[1] <= base_shake * 1.0001]
        err, shake, mc, be = min(ok)
        gain = 1 - err / base_err
        report[g] = {"bones": len(gb), "min_cutoff": mc, "beta": be,
                     "error_deg": {"none": round(raw_err, 3), "default": round(base_err, 3), "tuned": round(err, 3)},
                     "shake_when_still_deg": {"none": round(raw_shake, 4), "default": round(base_shake, 4),
                                              "tuned": round(shake, 4)}}
        print(f"{g:<8} ({len(gb):>2} bones)  error: none {raw_err:.2f}, default {base_err:.2f}, tuned {err:.2f} deg "
              f"({gain * 100:+.0f}%)   shake when still: none {raw_shake:.3f}, default {base_shake:.3f}, "
              f"tuned {shake:.3f} deg/frame  ->  min_cutoff {mc}, beta {be}")
        if gain >= args.min_gain:
            tuned[g] = {"min_cutoff": mc, "beta": be}
    if tuned:
        from avatarstudio.core.paths import save_json
        save_json(PATH, {"created": time.strftime("%Y-%m-%d %H:%M"), "source": args.video, "groups": tuned,
                         "report": report}, indent=1)
        print(f"saved -> {PATH} (groups: {', '.join(tuned)})")
    else:
        print("defaults are already as good: nothing saved")


if __name__ == "__main__":
    main()
