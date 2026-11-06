"""WiLoR finger helper: runs in its own environment (.venv_wilor, PyTorch + WiLoR-mini) and answers the studio.

Started by wilor_client.py; listens on 127.0.0.1:PORT. Per request the studio sends a small image patch around
each hand plus the hand's box inside the patch and whether it's a right hand; the reply is WiLoR's 21 3D hand
points (metres, camera axes, MediaPipe's point order) per hand.

Wire format (little endian): request = u32 n, then per hand: f32 x0 y0 x1 y1, f32 is_right, u32 h, u32 w,
h*w*3 bytes RGB. Reply = u32 n, then per hand: u8 ok, 63 x f32.
"""
import socket
import struct
import sys
import time

import numpy as np

PORT = 8766


def watch_parent(pid):
    """Exit as soon as the studio process `pid` ends, however it ended."""
    import os
    import threading
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not h:
            os._exit(0)  # already gone
        wait = lambda: k32.WaitForSingleObject(h, 0xFFFFFFFF)
    else:
        def wait():
            while True:
                try:
                    os.kill(pid, 0)
                except OSError:
                    return
                time.sleep(1)

    def run():
        wait()
        os._exit(0)
    threading.Thread(target=run, daemon=True).start()


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return bytes(buf)


def predict(pipe, hands, torch):
    """All hands in one batched pass. Same as WiLoR-mini's predict_with_bboxes, but the anti-aliasing blur is
    OpenCV's (on the small patch) instead of scikit-image's on a full image copy per hand, which made it ~2x
    slower. hands: [(box x0 y0 x1 y1 in the patch, is_right, RGB patch)] -> [21x3 points or None]."""
    import cv2
    from wilor_mini.utils import utils
    size = pipe.IMAGE_SIZE
    crops = []
    for (x0, y0, x1, y1), right, img in hands:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        box = 2.5 * max(x1 - x0, y1 - y0)
        down = box / size / 2.0
        if down > 1.1:
            sigma = (down - 1) / 2
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        patch, _ = utils.generate_image_patch_cv2(img.astype(np.float32), cx, cy, box, box, size, size,
                                                  right == 0, 1.0, 0, border_mode=cv2.BORDER_CONSTANT)
        crops.append(patch)
    with torch.no_grad():
        x = torch.from_numpy(np.stack(crops)).to(device=pipe.device, dtype=pipe.dtype)
        out = pipe.wilor_model(x)
    kp = out["pred_keypoints_3d"].float().cpu().numpy()
    res = []
    for i, (_, right, _) in enumerate(hands):
        k = kp[i].reshape(21, 3).copy()
        if right == 0:  # left hands are run mirrored
            k[:, 0] = -k[:, 0]
        res.append(k.astype(np.float32))
    return res


def main():
    if len(sys.argv) > 2:
        watch_parent(int(sys.argv[2]))
    try:  # a background helper gets throttled by Windows like the studio did (perf.py)
        from perf import boost_process
        boost_process()
    except Exception:
        pass
    import os
    # Apple GPUs (Metal, "mps"): an operation PyTorch can't run there yet goes to the CPU instead of failing
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    times = []
    # NVIDIA (CUDA; AMD's ROCm builds of PyTorch use the same name), Apple's Metal GPU, else the CPU. Half precision
    # only on CUDA, where it was measured; on Metal it's untested, so full precision there for now
    mps = getattr(torch.backends, "mps", None)
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if mps and mps.is_available() else "cpu")
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=torch.float16 if dev.type == "cuda" else torch.float32,
                                             verbose=False)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    print(f"WiLoR ready on {port} ({dev})", flush=True)
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                n, = struct.unpack("<I", recv_exact(conn, 4))
                t0 = time.perf_counter()
                hands = []
                for _ in range(n):
                    x0, y0, x1, y1, right = struct.unpack("<5f", recv_exact(conn, 20))
                    h, w = struct.unpack("<II", recv_exact(conn, 8))
                    patch = np.frombuffer(recv_exact(conn, h * w * 3), np.uint8).reshape(h, w, 3)
                    hands.append(((x0, y0, x1, y1), right, patch))
                kps = [None] * n
                try:
                    kps = predict(pipe, hands, torch)
                except Exception as e:
                    print(f"WiLoR error: {e}", flush=True)
                reply = [struct.pack("<I", n)]
                for kp in kps:
                    reply.append(struct.pack("<B", kp is not None))
                    reply.append((kp if kp is not None else np.zeros((21, 3), np.float32)).tobytes())
                conn.sendall(b"".join(reply))
                times.append((time.perf_counter() - t0, n))
                if len(times) == 100:
                    per = [t / max(k, 1) for t, k in times]
                    print(f"timing: {np.median([t for t, _ in times]) * 1000:.0f} ms per request, "
                          f"{np.median(per) * 1000:.0f} ms per hand", flush=True)
                    times.clear()
        except (ConnectionError, OSError):
            conn.close()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
