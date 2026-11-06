"""Small personal image classifiers on crops of the camera picture (tongue out, cheek puff, funnel).

A fixed pretrained feature extractor (models/backbone.onnx, see export_backbone.py) turns each crop into 960
numbers; a logistic regression learned from your own short recording separates the classes. Training takes
seconds and needs no PyTorch or GPU, so it also runs in the packaged app. Inference is plain numpy.

Validation is honest about time: neighbouring frames are near-copies, so a random split would look far better
than reality. A class with 2+ takes is checked on its newest whole take; with one take, on its last quarter.
"""
import json
import os
import threading
import time

import cv2
import numpy as np

BACKBONE_PATH = "models/backbone.onnx"
SIZE = 160


def square_crop(frame, cx, cy, size, angle_deg=0.0, out=SIZE):
    """size x size square around (cx, cy), rotated by angle_deg (to undo head tilt), resized to out x out RGB."""
    s = out / max(size, 1.0)
    m = cv2.getRotationMatrix2D((float(cx), float(cy)), angle_deg, s)
    m[0, 2] += out / 2 - cx
    m[1, 2] += out / 2 - cy
    crop = cv2.warpAffine(frame, m, (out, out), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def augment(crop, rng, flip=True):
    """A randomly varied copy: mirror, small turn/zoom/shift, brightness, contrast, colour, blur."""
    h, w = crop.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-15, 15), rng.uniform(0.85, 1.15))
    m[:, 2] += rng.uniform(-0.08, 0.08, 2) * w
    x = cv2.warpAffine(crop, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    if flip and rng.random() < 0.5:
        x = x[:, ::-1]
    x = x.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
    x = (x - x.mean()) * rng.uniform(0.8, 1.2) + x.mean()
    x *= rng.uniform(0.9, 1.1, 3)
    if rng.random() < 0.3:
        x = cv2.GaussianBlur(x, (5, 5), rng.uniform(0.5, 1.5))
    return np.clip(x, 0, 255).astype(np.uint8)


class Backbone:
    def __init__(self, path=BACKBONE_PATH):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

    def __call__(self, crops, batch=64):
        """crops: list / array of RGB uint8 160x160 -> N x 960 float32."""
        out = []
        for i in range(0, len(crops), batch):
            x = np.stack(crops[i:i + batch]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
            out.append(self.sess.run(None, {"image": x})[0])
        return np.concatenate(out) if out else np.zeros((0, 960), np.float32)


class Classifier:
    """Standardise + multinomial logistic regression, stored as plain arrays."""

    def __init__(self, classes, mean, std, coef, bias):
        self.classes, self.mean, self.std, self.coef, self.bias = list(classes), mean, std, coef, bias

    @classmethod
    def fit(cls, feats, labels, classes, c=0.5):
        from sklearn.linear_model import LogisticRegression
        mean, std = feats.mean(0), feats.std(0) + 1e-6
        lr = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
        lr.fit((feats - mean) / std, labels)
        coef = np.zeros((len(classes), feats.shape[1]), np.float32)
        bias = np.full(len(classes), -1e4, np.float32)  # classes with no data can never win
        seen = lr.classes_
        if len(seen) == 2:  # sklearn stores one row for two classes: split it so softmax = its sigmoid
            w, b = lr.coef_[0] / 2, lr.intercept_[0] / 2
            coef[seen[1]], bias[seen[1]], coef[seen[0]], bias[seen[0]] = w, b, -w, -b
        else:
            for row, k in enumerate(seen):
                coef[k], bias[k] = lr.coef_[row], lr.intercept_[row]
        return cls(classes, mean.astype(np.float32), std.astype(np.float32), coef, bias)

    def probs(self, feats):
        z = ((feats - self.mean) / self.std) @ self.coef.T + self.bias
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)

    def save(self, path, report):
        np.savez(path, mean=self.mean, std=self.std, coef=self.coef, bias=self.bias,
                 info=json.dumps({"classes": self.classes, "report": report, "created": time.strftime("%Y-%m-%d %H:%M")}))

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            return None
        d = np.load(path)
        info = json.loads(str(d["info"]))
        c = cls(info["classes"], d["mean"], d["std"], d["coef"], d["bias"])
        c.report = info.get("report")
        return c


def time_split(groups, labels=None):
    """Boolean mask of validation samples. A class recorded in 2+ takes (groups) is checked on its newest whole
    take, which the model never saw (the honest test); a class with one take on that take's last quarter."""
    val = np.zeros(len(groups), bool)
    labels = np.zeros(len(groups), int) if labels is None else np.asarray(labels)
    for k in np.unique(labels):
        in_k = labels == k
        takes = list(dict.fromkeys(groups[in_k]))  # in recording order
        if len(takes) >= 2:
            val[in_k & (groups == takes[-1])] = True
        else:
            idx = np.flatnonzero(in_k)
            val[idx[int(len(idx) * 0.75):]] = True
    return val


def train(crops, labels, groups, classes, backbone=None, augment_copies=3, flip=True, seed=0):
    """crops: N RGB 160x160; labels: class index; groups: segment id. Returns (Classifier on all data, report)."""
    rng = np.random.default_rng(seed)
    backbone = backbone or Backbone()
    labels, groups = np.asarray(labels), np.asarray(groups)
    val = time_split(groups, labels)

    def featurize(idx):
        imgs, ys = [], []
        for i in idx:
            imgs.append(crops[i])
            ys.append(labels[i])
            for _ in range(augment_copies):
                imgs.append(augment(crops[i], rng, flip))
                ys.append(labels[i])
        return backbone(imgs), np.array(ys)

    tr_idx, va_idx = np.flatnonzero(~val), np.flatnonzero(val)
    f_tr, y_tr = featurize(tr_idx)
    f_va = backbone([crops[i] for i in va_idx])
    clf = Classifier.fit(f_tr, y_tr, classes)
    pred = clf.probs(f_va).argmax(1) if len(va_idx) else np.zeros(0, int)
    y_va = labels[va_idx]
    report = {"val_accuracy": float(np.mean(pred == y_va)) if len(y_va) else float("nan"),
              "per_class": {}, "precision": {}, "confused_with": {}, "samples": {}}
    for k, name in enumerate(classes):
        m = y_va == k
        report["samples"][name] = int((labels == k).sum())
        if (pred == k).any():  # of everything recognised as this class, how much really was
            report["precision"][name] = float(np.mean(y_va[pred == k] == k))
        if m.any():
            report["per_class"][name] = float(np.mean(pred[m] == k))
            wrong = pred[m][pred[m] != k]
            if len(wrong):
                top = np.bincount(wrong, minlength=len(classes)).argmax()
                report["confused_with"][name] = classes[top]
    # final model on everything (train + held-out)
    f_all, y_all = featurize(np.arange(len(labels)))
    return Classifier.fit(f_all, y_all, classes), report


class AsyncRunner:
    """Runs a slow-changing classifier off the main loop: submit() hands it the newest frame when it is free
    (otherwise the frame is skipped), and .result is whatever finished last."""

    def __init__(self, fn):
        self.fn, self.result, self.busy = fn, None, False
        self.lock = threading.Lock()

    def submit(self, *args):
        with self.lock:
            if self.busy:
                return
            self.busy = True
        threading.Thread(target=self._run, args=args, daemon=True).start()

    def _run(self, *args):
        try:
            self.result = self.fn(*args)
        except Exception as e:
            print(f"{getattr(self.fn, '__name__', 'classifier')} error: {e}")
        finally:
            self.busy = False
