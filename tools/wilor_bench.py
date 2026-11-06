"""Try WiLoR (3D hand reconstruction, CVPR 2025) on a recording: speed on this GPU + its finger poses.

Runs in its own environment so it can't break the app's:
  .venv_wilor\\Scripts\\python.exe tools/wilor_bench.py take.mp4

WiLoR-mini downloads its weights (and the MANO hand model) from Hugging Face on first use. Two timings:
- full: its own YOLO hand detector + reconstruction
- boxes: reconstruction only, with hand boxes from MediaPipe (what the studio would do; MediaPipe already
  finds the hands every frame)
Saves wilor_take.npz: per frame and hand, MANO joint rotations (hand_pose 15x3x3, global_orient), is_right,
3D keypoints, box - for comparing finger shapes with the current solver (wilor_compare.py, main env).
"""
import argparse
import sys
import time

import cv2
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--boxes", help="npz with MediaPipe hand boxes per frame (from wilor_compare.py --export-boxes)")
    ap.add_argument("--out", default="wilor_take.npz")
    args = ap.parse_args()
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    print(f"device {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}), {dtype}", flush=True)
    t0 = time.time()
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=dtype, verbose=False)
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    boxes = np.load(args.boxes, allow_pickle=True)["boxes"] if args.boxes else None

    cap = cv2.VideoCapture(args.video)
    rows, t_full, t_box, i = [], [], [], 0
    while i < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        a = time.perf_counter()
        outs = pipe.predict(rgb)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_full.append(time.perf_counter() - a)
        if boxes is not None and i < len(boxes) and len(boxes[i]):
            b = np.array([x[:4] for x in boxes[i]], np.float32)
            right = np.array([x[4] for x in boxes[i]], np.float32)
            a = time.perf_counter()
            pipe.predict_with_bboxes(rgb, b, right)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_box.append(time.perf_counter() - a)
        for o in outs:
            p = o.get("wilor_preds")
            if not p:
                continue
            rows.append({"frame": i, "is_right": float(o["is_right"]), "box": np.array(o["hand_bbox"], np.float32),
                         "hand_pose": np.asarray(p["hand_pose"][0], np.float32),
                         "global_orient": np.asarray(p["global_orient"][0], np.float32),
                         "kp3d": np.asarray(p["pred_keypoints_3d"][0], np.float32)})
        i += 1
    ms = lambda t: (np.median(t[10:]) * 1000, np.percentile(t[10:], 95) * 1000) if len(t) > 10 else (0, 0)
    print(f"{i} frames, {len(rows)} hands found")
    print(f"full (detector + reconstruction): median {ms(t_full)[0]:.1f} ms, 95th {ms(t_full)[1]:.1f} ms per frame")
    if t_box:
        print(f"reconstruction with given boxes: median {ms(t_box)[0]:.1f} ms, 95th {ms(t_box)[1]:.1f} ms per frame")
    np.savez(args.out, rows=np.array(rows, dtype=object))
    print(f"saved {args.out}; pose shape {rows[0]['hand_pose'].shape if rows else None}")


if __name__ == "__main__":
    sys.exit(main())
