"""GPU whole-body model (RTMW-x, OpenMMLab, COCO-WholeBody 133 points): which hand is left / right.

RTMW is a top-down 2D model trained on 14 combined datasets: body 0-16, feet 17-22, face 23-90, left hand 91-111,
right hand 112-132 ("left" = the person's left). It runs on the GPU through ONNX Runtime + DirectML (macOS: Core ML,
Linux: CUDA / ROCm / OpenVINO; os_support.py), else on the CPU. Its wrists
(9, 10) decide which hand is left / right when hands cross or come together, where MediaPipe's own models mix
them up (all_in_one.AllInOneTracker._assign_sides). Blending its hand / arm points into MediaPipe's was tried and
made fingers slightly less plausible, so it was removed. The person box comes from the previous frame, so this
runs at the same time as the MediaPipe models.
"""
import os

import cv2
import numpy as np

MODEL_PATH = "models/rtmw-x-wholebody-384x288.onnx"
INPUT_W, INPUT_H = 288, 384
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


class WholeBody:
    def __init__(self, path=MODEL_PATH):
        import onnxruntime as ort
        from avatarstudio.core.os_support import onnx_session
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ort.set_default_logger_severity(3)
        self.sess, gpu = onnx_session(path)
        self.on_gpu = gpu is not None

    @staticmethod
    def box_from(points, w, h):
        """Person box (cx, cy, bw, bh) around image points, padded 25% and shaped like the model input."""
        if points is None or len(points) == 0:
            return w / 2, h / 2, w, h
        x0, y0 = points.min(0)
        x1, y1 = points.max(0)
        cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 1.25, (y1 - y0) * 1.25
        aspect = INPUT_W / INPUT_H
        if bw / max(bh, 1) > aspect:
            bh = bw / aspect
        else:
            bw = bh * aspect
        return cx, cy, max(bw, 64), max(bh, 64)

    def __call__(self, frame_bgr, box):
        """133 (x, y) image points and their scores (0..1) for the person in `box`."""
        cx, cy, bw, bh = box
        src = np.float32([[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2], [cx - bw / 2, cy + bh / 2]])
        dst = np.float32([[0, 0], [INPUT_W, 0], [0, INPUT_H]])
        m = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(frame_bgr, m, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
        x = ((cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
        sx, sy = self.sess.run(None, {"input": x})
        # SimCC: one score per sub-pixel column/row (split ratio 2); the peak gives the position and confidence
        px, py = sx[0].argmax(1) / 2.0, sy[0].argmax(1) / 2.0
        # raw SimCC peaks run ~0..8: out-of-frame points scored 0.5-1.5 and clearly visible ones 3-8 on the
        # sample photos, so map 1.5 -> 0 and 4.5 -> 1 to get a usable confidence
        score = (np.minimum(sx[0].max(1), sy[0].max(1)) - 1.5) / 3.0
        pts = np.stack([px, py], 1)
        inv = cv2.invertAffineTransform(m)
        pts = pts @ inv[:, :2].T + inv[:, 2]
        return pts.astype(np.float32), np.clip(score, 0, 1).astype(np.float32)
