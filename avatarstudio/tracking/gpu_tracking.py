"""GPU tracking: MediaPipe's own body, face and hand landmark models, run outside MediaPipe so that the body and
face models can use the GPU (ONNX Runtime + DirectML; MediaPipe's own GPU delegate isn't implemented on Windows).
On macOS they run on Core ML (Metal GPU / Neural Engine), on Linux on CUDA, ROCm or OpenVINO (os_support.py).

The models are MediaPipe's, converted once by convert_landmark_models.py (models/mp_*.onnx). What MediaPipe does
around them is copied here: the crop around the face / body / hand from the previous frame's points, turning the
model's points back into the image, the body model's heatmap refinement and smoothing. Finding a face, body or hand
that isn't tracked yet is still MediaPipe's job (its landmarkers in IMAGE mode); see GpuTracker.

Where the models run: body + face (+ the palm detector while a hand is missing) as ONE GPU run per frame, the two
hands on the CPU at the same time (ONNX Runtime). WiLoR shares the GPU (CUDA, another process): while it is busy,
every GPU run waits its turn, so four separate runs (body, face, two hands) took 12.9 ms and this split 7.8 ms
(4.9 ms without WiLoR). When the GPU thread is busy (the whole-body model) the same models run on the CPU.

All DirectML work of the app goes through one thread (GpuWorker), the whole-body model (wholebody.py) included:
two DirectML sessions running at the same time on different threads hung the GPU (DXGI device removed). Making a
DirectML session (~1 s) holds Python's GIL, so the app's other threads stall meanwhile (once, at the start).

Studio replay of take.mp4 (RTX 4060 laptop, WiLoR + whole-body model on): tracking 20.4 -> 16.7 ms per frame, frame ->
avatar data median 24 -> 18 ms, 95th percentile 32 -> 35 ms; without the whole-body model 13.3 ms, with neither
10.8 ms. Hands found 99.7% of MediaPipe's, left/right the same for 99.7-99.9%.
"""
import itertools
import math
import queue
import threading
import time
from collections import namedtuple
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np

FRAME_MODEL = "models/mp_pose_face_palm.onnx"  # body + face landmarks and the palm detector: one GPU run a frame
HAND_MODEL = "models/mp_hand_landmarks.onnx"
BLENDSHAPES_MODEL = "models/mp_face_blendshapes.onnx"
POSE_SIZE, FACE_SIZE, HAND_SIZE, PALM_SIZE = 256, 256, 224, 192
# the 146 face points the blendshape model reads (MediaPipe's face_blendshapes_graph.cc)
BLENDSHAPE_POINTS = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
    81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153,
    154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251,
    263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318,
    321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386,
    387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477]
BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight"]
HAND_ROI_POINTS = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]  # wrist + the two lower joints of each finger

Category = namedtuple("Category", "category_name score")  # (what all_in_one reads from MediaPipe's categories)


class Result:
    """The fields of a MediaPipe result that all_in_one.py reads, holding arrays (pixels) instead of objects."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


# ---------------------------------------------------------------- the one GPU thread
class GpuWorker:
    """The one thread that does all DirectML work in the app. Jobs run one at a time; frame work goes before
    background work (the whole-body model), which only starts when no frame job is waiting."""

    def __init__(self):
        self._jobs = queue.PriorityQueue()
        self._order = itertools.count()
        self._running = None  # (background?, start time) of the job running now
        self._background_s = 0.02  # how long a background job takes (running average)
        threading.Thread(target=self._run, daemon=True, name="gpu worker").start()

    def submit(self, fn, *args, background=False):
        job = Future()
        self._jobs.put((int(background), next(self._order), fn, args, job))
        return job

    def wait(self):
        """About how long a frame job submitted now would wait (s): the rest of a running background job."""
        running = self._running
        if running is None or not running[0]:
            return 0.0
        elapsed = time.perf_counter() - running[1]
        return 1.0 if elapsed > 2 * self._background_s else max(0.0, self._background_s - elapsed)  # (1.0: loading)

    def _run(self):
        while True:
            background, _, fn, args, job = self._jobs.get()
            start = time.perf_counter()
            self._running = (background, start)
            if job.set_running_or_notify_cancel():
                try:
                    job.set_result(fn(*args))
                except BaseException as e:
                    job.set_exception(e)
            if background:  # (capped: loading a model takes ~1 s once)
                self._background_s += 0.2 * (min(time.perf_counter() - start, 0.1) - self._background_s)
            self._running = None


_worker, _worker_lock = None, threading.Lock()


def gpu_worker():
    """The app's GpuWorker (started on first use)."""
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = GpuWorker()
        return _worker


def gpu_session(path):
    """A GPU session: DirectML on Windows, Core ML / CUDA / ... elsewhere (call it on the GPU worker). Raises if
    there's no GPU runtime here."""
    import onnxruntime as ort
    from avatarstudio.core.os_support import onnx_session
    ort.set_default_logger_severity(3)
    so = ort.SessionOptions()
    so.enable_mem_pattern = False  # (DirectML needs these two)
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess, gpu = onnx_session(path, so)
    if gpu is None:
        raise RuntimeError("no GPU runtime (DirectML / Core ML / CUDA / ...)")
    return sess


def cpu_session(path, threads):
    """A CPU session whose threads sleep between runs (spinning ones slowed the others: 4 sessions run side by side)."""
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------- MediaPipe's geometry (pixels)
# A rect is (centre x, centre y, width, height, rotation in radians, clockwise in the image).
def norm_angle(a):
    return a - 2 * math.pi * math.floor((a + math.pi) / (2 * math.pi))


def transform_rect(cx, cy, w, h, rot, scale, shift_y=0.0):
    """RectTransformationCalculator with square_long: shift along the rect's own y axis, square, scale."""
    cx -= h * shift_y * math.sin(rot)
    cy += h * shift_y * math.cos(rot)
    side = max(w, h) * scale
    return cx, cy, side, side, rot


def crop(rgb, rect, size, border=cv2.BORDER_REPLICATE):
    """The rotated rect as a size x size crop, like MediaPipe's ImageToTensor (OpenCV version): perspective warp
    of the rect's corners, bilinear. -> (1, size, size, 3) uint8."""
    cx, cy, w, h, rot = rect
    src = cv2.boxPoints(((cx, cy), (w, h), math.degrees(rot)))
    dst = np.float32([[0, size], [0, 0], [size, 0], [size, size]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(rgb, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=border)[None]


def project(xy, rect, size):
    """Points in crop pixels -> image pixels (LandmarkProjectionCalculator)."""
    cx, cy, w, h, rot = rect
    x, y = xy[:, 0] / size - 0.5, xy[:, 1] / size - 0.5
    c, s = math.cos(rot), math.sin(rot)
    return np.stack([(c * x - s * y) * w + cx, (s * x + c * y) * h + cy], 1)


def rotate_world(p, rot):
    """World points (metres) turned back by the crop's rotation (WorldLandmarkProjectionCalculator)."""
    c, s = math.cos(rot), math.sin(rot)
    out = p.copy()
    out[:, 0], out[:, 1] = c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]
    return out


def hand_rect(pts):
    """Next frame's hand crop from this frame's hand points (HandLandmarksToRectCalculator + 2x, shifted up)."""
    p = pts[HAND_ROI_POINTS, :2].astype(np.float64)
    x0, y0 = p[0]
    x1, y1 = ((p[4] + p[8]) / 2 + p[6]) / 2  # between the index and ring knuckles, then toward the middle one
    rot = norm_angle(math.pi / 2 - math.atan2(-(y1 - y0), x1 - x0))
    c0 = (p.max(0) + p.min(0)) / 2
    rev = norm_angle(-rot)
    o = p - c0
    q = np.stack([o[:, 0] * math.cos(rev) - o[:, 1] * math.sin(rev),
                  o[:, 0] * math.sin(rev) + o[:, 1] * math.cos(rev)], 1)
    lo, hi = q.min(0), q.max(0)
    pc = (lo + hi) / 2
    cx = pc[0] * math.cos(rot) - pc[1] * math.sin(rot) + c0[0]
    cy = pc[0] * math.sin(rot) + pc[1] * math.cos(rot) + c0[1]
    w, h = hi - lo
    return transform_rect(cx, cy, w, h, rot, 2.0, shift_y=-0.1)


def face_rect(pts):
    """Next frame's face crop: the points' box, turned so the eye corners (33, 263) are level, 1.5x."""
    p = pts[:, :2].astype(np.float64)
    lo, hi = p.min(0), p.max(0)
    rot = norm_angle(-math.atan2(-(p[263, 1] - p[33, 1]), p[263, 0] - p[33, 0]))
    return transform_rect((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[0] - lo[0], hi[1] - lo[1], rot, 1.5)


def pose_rect(aux):
    """Next frame's body crop from the body model's two extra points (body centre, scale + rotation point)."""
    (x0, y0), (x1, y1) = aux[:2, :2].astype(np.float64)
    size = 2 * math.hypot(x1 - x0, y1 - y0)
    rot = norm_angle(math.pi / 2 - math.atan2(-(y1 - y0), x1 - x0))
    return transform_rect(x0, y0, size, size, rot, 1.25)


def rect_iou(a, b):
    """Overlap of two rects' axis-aligned boxes (rotation ignored, like MediaPipe's rect association)."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def refine_from_heatmap(xy, heatmap, kernel=7, min_conf=0.5):
    """RefineLandmarksFromHeatmapCalculator: points (normalised crop coordinates) moved to the confidence-weighted
    centre (sigmoid of the scores) of the kernel x kernel heatmap cells around them, where the heatmap is
    confident. heatmap: (64, 64, n) raw scores. All points at once."""
    hh, hw, n = heatmap.shape
    off = (kernel - 1) // 2
    col, row = np.trunc(xy[:, 0] * hw).astype(int), np.trunc(xy[:, 1] * hh).astype(int)  # (C++ int(): toward 0)
    inside = (col >= 0) & (col < hw) & (row >= 0) & (row < hh)
    steps = np.arange(-off, off + 1)
    rows, cols = row[:, None] + steps, col[:, None] + steps  # (n, kernel)
    ok = ((rows >= 0) & (rows < hh))[:, :, None] & ((cols >= 0) & (cols < hw))[:, None, :]
    vals = heatmap[np.clip(rows, 0, hh - 1)[:, :, None], np.clip(cols, 0, hw - 1)[:, None, :],
                   np.arange(n)[:, None, None]].astype(np.float64)
    conf = np.where(ok, 1 / (1 + np.exp(-vals)), 0.0)  # (n, kernel rows, kernel cols)
    total = conf.sum((1, 2))
    use = inside & (conf.max((1, 2)) >= min_conf) & (total > 0)
    out = xy.copy()
    safe = np.where(total > 0, total, 1.0)
    out[use, 0] = ((conf.sum(1) * cols).sum(1) / hw / safe)[use]
    out[use, 1] = ((conf.sum(2) * rows).sum(1) / hh / safe)[use]
    return out


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, np.float64)))


# ---------------------------------------------------------------- MediaPipe's palm detector (finding new hands)
def _palm_anchors():
    """SsdAnchorsCalculator for the 192x192 palm detector: strides 8, 16, 16, 16 -> 24x24 cells x 2 anchors and 12x12
    cells x 6, every anchor 1x1 at its cell's centre. -> (2016, 2) centres."""
    out = []
    for grid, per_cell in ((24, 2), (12, 6)):
        c = (np.arange(grid) + 0.5) / grid
        xx, yy = np.meshgrid(c, c)
        out.append(np.repeat(np.stack([xx.ravel(), yy.ravel()], 1), per_cell, axis=0))
    return np.concatenate(out)


PALM_ANCHORS = _palm_anchors()


def palm_input(rgb):
    """The whole frame letterboxed into the palm detector's 192x192 input (ImageToTensor, black bars), and the
    square (side, in pixels) it covers."""
    h, w = rgb.shape[:2]
    side = max(w, h)
    return crop(rgb, (w / 2, h / 2, side, side, 0.0), PALM_SIZE, cv2.BORDER_CONSTANT), side


def palm_rects(boxes, scores, w, h, min_score, max_palms):
    """Palm detector outputs -> [(hand crop (rect in pixels), its 7 palm points (pixels))], best first:
    TensorsToDetections (sigmoid scores, anchors), weighted non-max suppression (IoU 0.3), letterbox removal, the
    palm box turned so wrist (point 0) -> middle knuckle (point 2) points up, then 2.6x and shifted toward the
    fingers (palm_detection_detection_to_roi)."""
    s = sigmoid(np.clip(scores.reshape(-1), -100, 100))
    keep = np.nonzero(s >= min_score)[0]
    if not len(keep):
        return []
    r = boxes.reshape(-1, 18)[keep].astype(np.float64) / PALM_SIZE
    a = PALM_ANCHORS[keep]
    c = r[:, :2] + a
    box = np.concatenate([c - r[:, 2:4] / 2, c + r[:, 2:4] / 2], 1)  # xmin, ymin, xmax, ymax
    kp = r[:, 4:].reshape(-1, 7, 2) + a[:, None]
    score = s[keep]
    order = list(np.argsort(-score, kind="stable"))
    dets = []
    while order and len(dets) < max_palms:
        top = box[order[0]]
        iw = np.clip(np.minimum(box[order, 2], top[2]) - np.maximum(box[order, 0], top[0]), 0, None)
        ih = np.clip(np.minimum(box[order, 3], top[3]) - np.maximum(box[order, 1], top[1]), 0, None)
        inter = iw * ih
        area = (box[order, 2] - box[order, 0]) * (box[order, 3] - box[order, 1])
        iou = inter / (area + (top[2] - top[0]) * (top[3] - top[1]) - inter + 1e-12)
        group = [o for o, v in zip(order, iou) if v > 0.3]
        wts = score[group][:, None]
        dets.append(((box[group] * wts).sum(0) / wts.sum(), (kp[group] * wts[:, :, None]).sum(0) / wts.sum()))
        order = [o for o, v in zip(order, iou) if v <= 0.3]
    side = max(w, h)
    rects = []
    for b, k in dets:  # letterbox square -> image pixels
        b = (b.reshape(2, 2) - 0.5) * side + [w / 2, h / 2]
        k = (k - 0.5) * side + [w / 2, h / 2]
        rot = norm_angle(math.pi / 2 - math.atan2(-(k[2, 1] - k[0, 1]), k[2, 0] - k[0, 0]))
        (x0, y0), (x1, y1) = b
        rects.append((transform_rect((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, rot, 2.6, shift_y=-0.5), k))
    return rects


# ---------------------------------------------------------------- MediaPipe's smoothing (video mode)
def _alpha(cutoff, freq):
    return 1.0 / (1.0 + freq / (2 * math.pi * cutoff))


class OneEuro:
    """MediaPipe's OneEuroFilter, one per value (vectorised over all points and axes)."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0, freq=30.0):
        self.min_cutoff, self.beta, self.d_cutoff, self.freq0 = min_cutoff, beta, d_cutoff, freq
        self.reset()

    def reset(self):
        self.t, self.freq, self.x, self.raw, self.dx = None, self.freq0, None, None, None

    def __call__(self, v, t, scale=1.0):
        v = np.asarray(v, np.float64)
        if self.t is not None:
            if t <= self.t:
                return v
            self.freq = 1.0 / (t - self.t)
        self.t = t
        dv = np.zeros_like(v) if self.raw is None else (v - self.raw) * scale * self.freq
        a = _alpha(self.d_cutoff, self.freq)
        self.dx = dv if self.dx is None else a * dv + (1 - a) * self.dx
        a = _alpha(self.min_cutoff + self.beta * np.abs(self.dx), self.freq)
        self.x = v if self.x is None else a * v + (1 - a) * self.x
        self.raw = v
        return self.x


class LowPass:
    def __init__(self, alpha):
        self.alpha, self.x = alpha, None

    def reset(self):
        self.x = None

    def __call__(self, v):
        v = np.asarray(v, np.float64)
        self.x = v if self.x is None else self.alpha * v + (1 - self.alpha) * self.x
        return self.x


# ---------------------------------------------------------------- model outputs -> points
def pose_points(out, rect, heatmap_refine=True):
    """Body model outputs -> (39, 3) image points (z in x's scale), (39, 3) world points, (39,) visibility, presence."""
    lm, presence, heatmap, world = out
    a = lm.reshape(39, 5).astype(np.float64)
    xy = a[:, :2] / POSE_SIZE
    if heatmap_refine:
        xy = refine_from_heatmap(xy, heatmap[0])
    pts = np.empty((39, 3))
    pts[:, :2] = project(xy * POSE_SIZE, rect, POSE_SIZE)
    pts[:, 2] = a[:, 2] / POSE_SIZE * rect[2]
    return pts, rotate_world(world.reshape(39, 3).astype(np.float64), rect[4]), sigmoid(a[:, 3]), float(presence.ravel()[0])


def face_points(out, rect):
    """Face model outputs -> (478, 3) image points, presence."""
    lm, presence_logit = out
    a = lm.reshape(478, 3).astype(np.float64)
    pts = np.empty((478, 3))
    pts[:, :2] = project(a[:, :2], rect, FACE_SIZE)
    pts[:, 2] = a[:, 2] / FACE_SIZE * rect[2]
    return pts, float(sigmoid(presence_logit.ravel()[0]))


def hand_points(out, rect):
    """Hand model outputs -> (21, 3) image points, (21, 3) world points, presence, right-hand score."""
    lm, presence, handedness, world = out
    a = lm.reshape(21, 3).astype(np.float64)
    pts = np.empty((21, 3))
    pts[:, :2] = project(a[:, :2], rect, HAND_SIZE)
    pts[:, 2] = a[:, 2] / HAND_SIZE / 0.4 * rect[2]
    return (pts, rotate_world(world.reshape(21, 3).astype(np.float64), rect[4]), float(presence.ravel()[0]),
            float(handedness.ravel()[0]))


# ---------------------------------------------------------------- tracking
POSE_OUTPUTS = ["pose/landmarks", "pose/presence", "pose/heatmap", "pose/world"]
FACE_OUTPUTS = ["face/landmarks", "face/presence_logit"]
PALM_OUTPUTS = ["palm/boxes", "palm/scores"]
HAND_OUTPUTS = ["landmarks", "presence", "handedness", "world"]
DUPLICATE = 0.2  # two hand results whose points are this close on average (in hand sizes) are the same hand
NEW_PALM = 0.2   # a palm whose wrist + middle knuckle are further than this (hand sizes) from a hand's is a new hand
CPU_INSTEAD_S = 0.010  # the GPU thread is busy for longer than this (whole-body model): body + face on the CPU


def _mp_points(landmarks, w, h):
    """MediaPipe's normalised points -> pixels (z in x's scale), like all_in_one._np."""
    return np.array([(p.x * w, p.y * h, p.z * w) for p in landmarks])


def _hand_size(pts):
    return float(np.linalg.norm(pts[0, :2] - pts[9, :2]))  # wrist -> middle knuckle


def _same_hand(a, b):
    return np.linalg.norm(a[:, :2] - b[:, :2], axis=1).mean() < DUPLICATE * max(_hand_size(a), _hand_size(b))


class GpuTracker:
    """MediaPipe's video-mode tracking of one body, one face and up to two hands, with the landmark models run here.

    Copied from MediaPipe (and checked against it on take.mp4, crop from MediaPipe's previous frame -> its result):
    hands 0.0015 px, face 0.06 px, body points within the model's own frame-to-frame noise. Its video-mode
    smoothing is copied too: body one-euro (min cutoff 0.05, beta 80; world 0.1, 40), visibility low-pass 0.1, face
    one-euro (0.05, 80); hands aren't smoothed. The next body / face crop comes from the unsmoothed points.

    GPU or CPU: the body + face (+ palm) models run as one GPU run (~3 ms) when the GPU thread is free within
    CPU_INSTEAD_S, and otherwise as the same model on the CPU (~13 ms alone, ~21 ms with everything else running):
    while the whole-body model runs (17-30 ms, it shares the GPU thread), while the GPU session loads, after a GPU
    error. A frame never waits long for the GPU, and tracking starts before the GPU session is ready.

    Finding what isn't tracked is MediaPipe's own job (its landmarkers in IMAGE mode), and its result is used as it is
    for that frame, like MediaPipe's first frame after a detection. Face / body: when lost. Hands: MediaPipe searches
    (palm detector + the hand model on what it finds, ~18 ms on the CPU) on every frame with fewer hands than
    wanted - a third of take.mp4's frames, and only 1 in 10 of those found a hand. Here the palm detector runs in the
    frame's GPU run (decoding copied from MediaPipe), and MediaPipe's search runs when it sees a palm that isn't one
    of the tracked hands (NEW_PALM): 136 frames instead of 355 on take.mp4, including all 36 where a hand was found.
    Such palms come in runs of frames, so after one the next frame's search starts right away, alongside the rest.
    (Tried and dropped: taking new hands from the palm detector directly - 3% fewer hands, its crops couldn't be
    matched to MediaPipe's exactly and with hands held together they latched onto the tracked hand; skipping
    searches after fruitless ones - the hands found came at the end of runs of fruitless searches.)
    hand_search="always": search like MediaPipe."""

    def __init__(self, make_detectors, max_hands=2, min_presence=0.4, hand_search="palm"):
        """make_detectors() -> MediaPipe's (face, hand, pose) landmarkers in IMAGE mode."""
        self.blank = {"pose": np.zeros((1, POSE_SIZE, POSE_SIZE, 3), np.uint8),
                      "face": np.zeros((1, FACE_SIZE, FACE_SIZE, 3), np.uint8),
                      "palm": np.zeros((1, PALM_SIZE, PALM_SIZE, 3), np.uint8)}
        self.cpu = cpu_session(FRAME_MODEL, 4)
        self.face_detector, self.hand_detector, self.pose_detector = make_detectors()
        self.hand_models = [cpu_session(HAND_MODEL, 2) for _ in range(max_hands)]
        self.blendshape_model = cpu_session(BLENDSHAPES_MODEL, 1)
        self.worker, self.gpu, self.gpu_fails, self.gpu_reloads = gpu_worker(), None, 0, 0
        # the GPU session loads in the background (~1 s; made at the same time, the CPU sessions took 8x longer):
        self.worker.submit(self._load_gpu)  # frames run on the CPU until it's ready
        self.pool = ThreadPoolExecutor(2 * max_hands + 6, thread_name_prefix="gpu tracking")
        self.max_hands, self.min_presence, self.hand_search = max_hands, min_presence, hand_search
        self.face = self.body = None  # what's tracked: next frame's crop + smoothing filters
        self.hands = []
        self.new_palm = False  # last frame's palm detector saw a palm that isn't a tracked hand
        self.frames = {"gpu": 0, "cpu": 0}  # where the body + face models ran

    def _load_gpu(self):
        try:
            sess = gpu_session(FRAME_MODEL)
            sess.run(None, self.blank)  # (the first run sets the GPU up)
            self.gpu = sess
        except Exception as e:  # (no DirectML GPU, ...)
            print(f"GPU tracking: GPU unavailable ({str(e).splitlines()[0][:120]}); its models run on the CPU",
                  flush=True)

    def reload(self):
        """A new GPU session (after GPU errors); frames run on the CPU meanwhile."""
        self.gpu = None
        self.worker.submit(self._load_gpu)

    def _models(self, feed, names):
        """Start the frame model for the outputs `names` -> a function returning {name: array}: one GPU run if the GPU
        thread is free soon enough, else the same model on the CPU."""
        gpu = self.gpu
        if gpu is not None and self.worker.wait() < CPU_INSTEAD_S:
            self.frames["gpu"] += 1
            job = self.worker.submit(gpu.run, names, feed)
            return lambda: self._gpu_result(job, feed, names)
        self.frames["cpu"] += 1
        job = self.pool.submit(self.cpu.run, names, feed)  # (one run: parts side by side oversubscribed the CPU)
        return lambda: dict(zip(names, job.result()))

    def _gpu_result(self, job, feed, names):
        try:
            out = job.result()
            self.gpu_fails = 0
            return dict(zip(names, out))
        except Exception as e:  # (driver reset, GPU switched off, ...): this frame on the CPU
            self.gpu_fails += 1
            if self.gpu_fails == 1:
                print(f"GPU tracking: GPU error ({str(e).splitlines()[0][:120]}); on the CPU meanwhile", flush=True)
            if self.gpu_fails == 3:
                self.gpu = None
                if self.gpu_reloads < 2:  # rebuild the session (twice at most), else stay on the CPU
                    self.gpu_reloads += 1
                    self.reload()
            return dict(zip(names, self.cpu.run(names, feed)))

    def __call__(self, rgb, image, ts, face=True):
        """One video frame -> (face, body, hands) like MediaPipe's results, holding pixel arrays (see Result)."""
        h, w = rgb.shape[:2]
        t = ts / 1000
        hand_jobs = [(hand["rect"], self.pool.submit(self._hand, model, rgb, hand["rect"]))
                     for hand, model in zip(self.hands, self.hand_models)]  # hands: on the CPU, from last frame's crops
        find = {}  # MediaPipe's own searches, on the CPU at the same time as the rest
        if face and self.face is None:
            find["face"] = self.pool.submit(self.face_detector.detect, image)
        if self.body is None:
            find["body"] = self.pool.submit(self.pose_detector.detect, image)
        look = len(self.hands) < self.max_hands  # (MediaPipe looks for more hands on these frames)
        palm = look and self.hand_search == "palm"
        if look and (self.hand_search == "always" or palm and self.new_palm):
            find["hands"] = self.pool.submit(self.hand_detector.detect, image)
        body_rect = self.body["rect"] if self.body else None
        face_rect_ = self.face["rect"] if face and self.face else None
        names = (POSE_OUTPUTS if body_rect else []) + (FACE_OUTPUTS if face_rect_ else []) + (PALM_OUTPUTS if palm else [])
        models = None
        if names:  # body, face and palm detector
            models = self._models({"pose": crop(rgb, body_rect, POSE_SIZE) if body_rect else self.blank["pose"],
                                   "face": crop(rgb, face_rect_, FACE_SIZE) if face_rect_ else self.blank["face"],
                                   "palm": palm_input(rgb)[0] if palm else self.blank["palm"]}, names)

        fr = Result(face_landmarks=[], face_blendshapes=[])
        pr = Result(pose_landmarks=[], pose_world_landmarks=[], visibility=None)
        out = models() if models else {}
        self.new_palm = palm and self._new_palm(out, w, h)
        if self.new_palm and "hands" not in find:  # MediaPipe's own search, as soon as such a palm is seen
            find["hands"] = self.pool.submit(self.hand_detector.detect, image)
        if body_rect:
            self._track_body([out[n] for n in POSE_OUTPUTS], body_rect, t, pr)
        if face_rect_:
            self._track_face([out[n] for n in FACE_OUTPUTS], face_rect_, t, fr)
        hands = []
        for rect, job in hand_jobs:
            pts, world, presence, right = hand_points(job.result(), rect)
            if presence >= self.min_presence and not any(_same_hand(pts, k["pts"]) for k in hands):
                label = Category("Right", right) if right > 0.5 else Category("Left", 1 - right)
                hands.append({"pts": pts, "world": world, "label": label})

        if "body" in find:
            r = find["body"].result()
            if r.pose_landmarks:
                pts = _mp_points(r.pose_landmarks[0], w, h)
                world = np.array([(p.x, p.y, p.z) for p in r.pose_world_landmarks[0]])
                vis = np.array([p.visibility or 0 for p in r.pose_landmarks[0]])
                self.body = {"lm": OneEuro(0.05, 80.0), "world": OneEuro(0.1, 40.0), "vis": LowPass(0.1)}
                self._set_body(pr, self.body["lm"](pts, t), self.body["world"](world, t), self.body["vis"](vis))
                self.body["rect"] = self._body_rect_from(rgb, pts)
        if "face" in find:
            r = find["face"].result()
            if r.face_landmarks:
                pts = _mp_points(r.face_landmarks[0], w, h)
                self.face = {"filter": OneEuro(0.05, 80.0), "rect": face_rect(pts)}
                self.face["filter"](pts, t)
                fr.face_landmarks, fr.face_blendshapes = [pts.astype(np.float32)], r.face_blendshapes[:1]
        if "hands" in find:  # tracked hands come first: a found hand that is one of them is dropped
            r = find["hands"].result()
            for lms, world, hd in zip(r.hand_landmarks, r.hand_world_landmarks, r.handedness):
                pts = _mp_points(lms, w, h)
                if not any(_same_hand(pts, k["pts"]) for k in hands):
                    hands.append({"pts": pts, "world": np.array([(p.x, p.y, p.z) for p in world]),
                                  "label": Category(hd[0].category_name, hd[0].score)})
        hands = hands[:self.max_hands]
        self.hands = [{"rect": hand_rect(k["pts"]), "pts": k["pts"]} for k in hands]
        hr = Result(hand_landmarks=[k["pts"].astype(np.float32) for k in hands],
                    hand_world_landmarks=[k["world"].astype(np.float32) for k in hands],
                    handedness=[[k["label"]] for k in hands])
        return fr, pr, hr

    def _hand(self, model, rgb, rect):
        return model.run(HAND_OUTPUTS, {"image": crop(rgb, rect, HAND_SIZE)})

    def _new_palm(self, out, w, h):
        """Does the palm detector see a palm that isn't one of the tracked hands (as of last frame)?"""
        return any(all(np.linalg.norm(kp[[0, 2]] - hand["pts"][[0, 9], :2], axis=1).mean() >
                       NEW_PALM * _hand_size(hand["pts"]) for hand in self.hands)
                   for _, kp in palm_rects(out["palm/boxes"], out["palm/scores"], w, h, self.min_presence,
                                           self.max_hands))

    def _track_body(self, out, rect, t, pr):
        pts, world, vis, presence = pose_points(out, rect)
        if presence < self.min_presence:
            self.body = None  # lost: MediaPipe's search from the next frame
            return
        b = self.body
        self._set_body(pr, b["lm"](pts[:33], t, 1 / rect[2]), b["world"](world[:33], t), b["vis"](vis[:33]))
        b["rect"] = pose_rect(pts[33:35])

    @staticmethod
    def _set_body(pr, pts, world, vis):
        pr.pose_landmarks, pr.pose_world_landmarks = [pts.astype(np.float32)], [world.astype(np.float32)]
        pr.visibility = vis.astype(np.float32)

    def _track_face(self, out, rect, t, fr):
        pts, presence = face_points(out, rect)
        if presence < self.min_presence:
            self.face = None
            return
        lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
        smooth = self.face["filter"](pts, t, 2 / max((hi - lo).sum(), 1e-6))
        self.face["rect"] = face_rect(pts)
        scores = self.blendshape_model.run(None, {"points": smooth[BLENDSHAPE_POINTS, :2].astype(np.float32)[None]})[0]
        fr.face_landmarks = [smooth.astype(np.float32)]
        fr.face_blendshapes = [[Category(n, float(s)) for n, s in zip(BLENDSHAPE_NAMES, scores.ravel())]]

    def _body_rect_from(self, rgb, pts):
        """Next frame's body crop after MediaPipe found the body. Its result lacks the model's two extra points that
        the crop comes from, so the body model runs twice more on crops guessed from the points (centred on the
        hips, up toward the shoulders) and its extra points from there give the crop."""
        hip, shoulders = (pts[23, :2] + pts[24, :2]) / 2, (pts[11, :2] + pts[12, :2]) / 2
        up = shoulders - hip
        reach = 1.1 * np.linalg.norm(pts[:33, :2] - hip, axis=1).max()
        rect = pose_rect(np.array([hip, hip + up / (np.linalg.norm(up) + 1e-9) * reach]))
        for _ in range(2):
            out = self._models({**self.blank, "pose": crop(rgb, rect, POSE_SIZE)}, POSE_OUTPUTS)()
            rect = pose_rect(pose_points([out[n] for n in POSE_OUTPUTS], rect)[0][33:35])
        return rect


