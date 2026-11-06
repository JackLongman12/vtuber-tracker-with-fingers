"""Tongue out, cheek puff and lip funnel from a picture of your mouth.

MediaPipe has no tongue at all, and its cheekPuff / mouthFunnel barely move for many people. This looks at a
level, head-tilt-corrected crop of the mouth and cheeks with a small personal classifier (crop_model.py),
trained from the extra prompts in the face recording ("Train Face.bat").

A shape is only switched on if it was recognised reliably on held-out moments of your recording (at least 60%
found and 80% of its detections right); weak ones stay off rather than firing randomly on stream.
The values are sent as the ARKit shapes tongueOut / cheekPuff / mouthFunnel (Perfect Sync avatars) and, for
VRM avatars, a custom "TongueOut" clip if your model has one.
"""
import os

import numpy as np

from avatarstudio.tracking.crop_model import AsyncRunner, Backbone, Classifier, square_crop, train
from avatarstudio.core.paths import user_file

MODEL_PATH = user_file("mouth_model.npz")
DATA_PATH = user_file("mouth_data.npz")
SHAPES = ["tongueOut", "cheekPuff", "mouthFunnel"]
CLASSES = ["none"] + SHAPES
MIN_RECALL, MIN_PRECISION = 0.6, 0.8


def mouth_crop(frame, face):
    """Square over mouth, chin and cheeks, rotated so the mouth corners are level."""
    p = face["_pts"][:, :2]
    a, b = p[61], p[291]  # mouth corners
    d = b - a
    width = float(np.linalg.norm(d))
    angle = float(np.degrees(np.arctan2(d[1], d[0])))
    down = np.array([-d[1], d[0]]) / max(width, 1e-6)  # perpendicular to the mouth line, toward the chin
    if np.dot(down, p[152] - p[1]) < 0:  # (chin minus nose tip points down the face)
        down = -down
    c = (p[13] + p[14]) / 2 + 0.25 * width * down  # a bit below the lips: room for a tongue
    return square_crop(frame, c[0], c[1], 2.4 * width, angle)


class MouthModel:
    def __init__(self, path=MODEL_PATH):
        self.clf = Classifier.load(path)
        if self.clf is None:
            raise FileNotFoundError(path)
        self.enabled = (getattr(self.clf, "report", None) or {}).get("enabled", [])
        self.backbone = Backbone()
        self.runner = AsyncRunner(self._classify)
        self.smooth, self._last = None, None

    def _classify(self, crop):
        return self.clf.probs(self.backbone([crop]))[0]

    def __call__(self, frame, face):
        """-> {ARKit shape: 0..1} for the enabled shapes (one frame behind; they're held for a while anyway)."""
        if face is None or not self.enabled:
            self.smooth = None
            return {}
        self.runner.submit(mouth_crop(frame, face))
        r = self.runner.result
        if r is not None and r is not self._last:
            self._last = r
            self.smooth = r if self.smooth is None else 0.5 * self.smooth + 0.5 * r
        if self.smooth is None:
            return {}
        return {s: float(np.clip((self.smooth[CLASSES.index(s)] - 0.5) / 0.4, 0, 1)) for s in self.enabled}


def train_and_save(crops, labels, groups, path=MODEL_PATH):
    """Returns the report; saves only if at least one shape is reliable."""
    labels = np.asarray(labels)
    if len(crops) < 100 or not any((labels == k).sum() >= 20 for k in range(1, len(CLASSES))):
        return None
    clf, report = train(np.asarray(crops), labels, groups, CLASSES)
    report["enabled"] = [s for s in SHAPES if report["per_class"].get(s, 0) >= MIN_RECALL
                         and report["precision"].get(s, 0) >= MIN_PRECISION]
    if report["enabled"]:
        clf.save(path, report)
    elif os.path.exists(path):
        os.remove(path)  # an older model for a different face / light would be worse than none
    return report
