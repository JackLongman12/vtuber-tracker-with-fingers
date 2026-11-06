"""Finger shapes from WiLoR (CVPR 2025 3D hand reconstruction) instead of MediaPipe's finger points.

WiLoR needs PyTorch, so it runs as a helper process (wilor_server.py) in its own environment, .venv_wilor (see
README "WiLoR fingers"); the studio and the packaged builds stay PyTorch-free. Every frame the studio hands it
small patches around the hands MediaPipe found; WiLoR answers in ~40 ms on an RTX 4060 laptop, so results come
in for every 2nd-3rd frame and the newest one is used. Only the finger shapes come from WiLoR - the wrist and
hand orientation stay with MediaPipe's newest frame, so WiLoR's delay only affects the fingers. If the helper
isn't installed, is still starting or falls behind (> FRESH_S), fingers come from MediaPipe as before.

Measured on a recorded take, same rig: finger shake 0.484 -> 0.371 deg/frame, single-frame finger glitches
78 -> 55, wrist glitches 6 -> 0 (WiLoR refreshed every 2nd frame); open hands read straight, fists curl more.
"""
import collections
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

PORT = 8766
FRESH_S = 0.25      # use a WiLoR result for this long after its frame
MAX_PATCH = 512     # patches larger than this are scaled down before sending
MAX_RESTARTS = 2    # the helper is restarted this often if it dies


def _find(rel_paths):
    bases = [os.getcwd(), os.path.dirname(os.path.abspath(__file__)),
             os.path.join(os.getcwd(), "..", ".."),  # a packaged build inside <project>/Distributions/
             os.path.join(os.path.dirname(sys.executable), "..", "..")]
    exe = os.environ.get("AVATARSTUDIO_EXE")  # the one-file app (it runs unpacked from %LOCALAPPDATA%)
    if exe:
        bases += [os.path.dirname(exe), os.path.join(os.path.dirname(exe), "..")]  # <project>/Distributions/x.exe
    for b in bases:
        for r in rel_paths:
            p = os.path.abspath(os.path.join(b, r))
            if os.path.exists(p):
                return p
    return None


def find_wilor():
    """(python.exe of the WiLoR environment, wilor_server.py) or (None, None)."""
    from avatarstudio.core.os_support import venv_python
    py = os.environ.get("AVATARSTUDIO_WILOR") or _find([venv_python(".venv_wilor")])
    script = _find(["wilor_server.py"])
    return (py, script) if py and script else (None, None)


class WilorFingers:
    def __init__(self, port=PORT):
        py, script = find_wilor()
        if not py:
            raise FileNotFoundError("WiLoR environment (.venv_wilor) or wilor_server.py not found")
        self.port, self.state, self.cmd, self.restarts = port, "starting", (py, script), 0
        self._start()
        self.results = {}  # side -> (time, 21x3 points)
        self.round_trips = []  # seconds from frame to result, recent requests
        self.job, self.busy, self.done = None, False, 0  # done: requests finished (see wait())
        self.cond = threading.Condition()
        self.running = True
        threading.Thread(target=self._log, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()

    def _start(self):
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        py, script = self.cmd
        # run the helper from a copy in %TEMP%: .venv_wilor is usually built on the Microsoft Store Python, which
        # sees a private, virtualised %LOCALAPPDATA% - where the one-file app unpacks itself - so it couldn't open
        # the script there ("No such file"). The temp folder isn't virtualised.
        run_dir = os.path.join(tempfile.gettempdir(), "AvatarStudio-wilor")
        try:
            os.makedirs(run_dir, exist_ok=True)
            # perf.py goes along (the helper imports it): next to the script in the packaged app, in core/ here
            perf = os.path.join(os.path.dirname(script), "perf.py")
            if not os.path.exists(perf):
                perf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "perf.py")
            for src in (script, perf):
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(run_dir, os.path.basename(src)))
            script = os.path.join(run_dir, "wilor_server.py")
        except OSError:  # (read-only temp?) try where it is
            run_dir = os.path.dirname(script)
        # our pid: the helper quits when we're gone (a venv python.exe is a launcher, so terminate() alone
        # can leave the real interpreter running on the GPU)
        self.proc = subprocess.Popen([py, script, str(self.port), str(os.getpid())], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, creationflags=flags, cwd=run_dir)

    def _log(self):
        """Reads the helper's messages; if it dies while we run (GPU driver reset, ...), restarts it - twice at
        most, then fingers stay with MediaPipe."""
        while True:
            tail = collections.deque(maxlen=3)  # the helper's last words, shown if it dies
            for line in self.proc.stdout:
                line = line.strip()
                if line and "Warning" not in line and "warn" not in line:
                    tail.append(line)
                if "WiLoR ready" in line:
                    self.state = "running"
                elif line.startswith("timing:"):
                    print(f"WiLoR {line}", flush=True)
                elif line and ("Error" in line or "error" in line) and "Warning" not in line:
                    print(f"WiLoR: {line}", flush=True)
            if not self.running:
                return
            self.state = "failed"
            for line in tail:
                print(f"WiLoR: {line[:300]}", flush=True)
            if self.restarts >= MAX_RESTARTS:
                print("WiLoR helper stopped; fingers come from MediaPipe", flush=True)
                return
            self.restarts += 1
            print(f"WiLoR helper stopped; restarting it ({self.restarts}/{MAX_RESTARTS}), MediaPipe fingers meanwhile",
                  flush=True)
            time.sleep(2.0)
            if not self.running:
                return
            try:
                self.state = "starting"
                self._start()
            except OSError as e:
                print(f"WiLoR restart failed ({e}); fingers come from MediaPipe", flush=True)
                self.state = "failed"
                return

    def submit(self, frame, hands):
        """Queue this frame's hands (the newest replaces one still waiting). Only the frame and the hands'
        points are handed over: cutting the patches out happens on the worker thread, off the tracking loop
        (the frame must not be drawn on afterwards - the studio window draws on a copy)."""
        if self.state != "running" or not hands:
            return
        with self.cond:
            self.job = (time.perf_counter(), frame, [(h["side"], h["_pts"][:, :2].copy()) for h in hands])
            self.cond.notify()

    @staticmethod
    def _patches(frame, hands):
        h, w = frame.shape[:2]
        patches = []
        for side, p in hands:
            lo, hi = p.min(0), p.max(0)
            c, s = (lo + hi) / 2, float((hi - lo).max())
            half = max(1.6 * s, 40.0)  # WiLoR looks at 2.5x the box: give it that much room
            x0, y0 = int(max(c[0] - half, 0)), int(max(c[1] - half, 0))
            x1, y1 = int(min(c[0] + half, w)), int(min(c[1] + half, h))
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            patch = frame[y0:y1, x0:x1]
            k = min(1.0, MAX_PATCH / max(patch.shape[:2]))
            if k < 1.0:
                patch = cv2.resize(patch, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            box = ((lo - [x0, y0]) * k).tolist() + ((hi - [x0, y0]) * k).tolist()
            patches.append((side, box, cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)))
        return patches

    def _worker(self):
        sock = None
        while self.running:
            with self.cond:
                self.cond.wait_for(lambda: self.job is not None or not self.running, timeout=0.5)
                job, self.job = self.job, None
            if job is None or self.state != "running":
                continue
            try:
                sock = self._run(job, sock)
            finally:
                with self.cond:  # (mocap.py waits for every frame's fingers)
                    self.done += 1
                    self.cond.notify_all()

    def _run(self, job, sock):
        """One request to the helper; returns the connection to reuse (None after a failure)."""
        t, frame, hands = job
        patches = self._patches(frame, hands)
        if not patches:
            return sock
        try:
            if sock is None:
                sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            msg = [struct.pack("<I", len(patches))]
            for side, box, img in patches:
                msg.append(struct.pack("<5f", *box, 1.0 if side == "right" else 0.0))
                msg.append(struct.pack("<II", img.shape[0], img.shape[1]))
                msg.append(np.ascontiguousarray(img).tobytes())
            sock.sendall(b"".join(msg))
            n, = struct.unpack("<I", self._recv(sock, 4))
            for side, _, _ in patches[:n]:
                ok, = struct.unpack("<B", self._recv(sock, 1))
                kp = np.frombuffer(self._recv(sock, 63 * 4), np.float32).reshape(21, 3)
                if ok:
                    self.results[side] = (t, kp - kp.mean(0))
            self.round_trips.append(time.perf_counter() - t)
            del self.round_trips[:-300]
            return sock
        except (OSError, struct.error):
            if sock is not None:
                sock.close()
            time.sleep(0.2)
            return None

    def wait(self, n, timeout=2.0):
        """Wait until `n` requests are done (or the helper stops): offline use, where every frame's fingers count."""
        with self.cond:
            self.cond.wait_for(lambda: self.done >= n or self.state != "running" or not self.running, timeout)

    @staticmethod
    def _recv(sock, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise OSError("closed")
            buf += chunk
        return bytes(buf)

    def apply(self, res):
        """Give each hand WiLoR's finger points if a fresh enough result exists (hand['_world_fingers'])."""
        now = time.perf_counter()
        for hand in res["hands"]:
            if "_cam" in hand:  # only seen by an extra camera: WiLoR's crops come from the main camera
                continue
            r = self.results.get(hand["side"])
            if r is not None and now - r[0] < FRESH_S:
                hand["_world_fingers"] = r[1]

    def summary(self):
        r = np.array(self.round_trips) * 1000
        return (f"WiLoR: {len(r)} results, frame -> fingers median {np.median(r):.0f} ms, 95th "
                f"{np.percentile(r, 95):.0f} ms") if len(r) else "WiLoR: no results"

    def close(self):
        self.running = False
        with self.cond:
            self.cond.notify()
        try:
            self.proc.terminate()
        except OSError:
            pass
