"""Webcam reader on a background thread: the tracker always gets the newest frame instead of waiting for
the camera, and stale frames are dropped, so speed = the slower of camera and tracking, not their sum.
If the camera disappears (unplugged, taken by another app) it keeps trying to reopen it."""
import sys
import threading
import time

import cv2


class CameraStream:
    def __init__(self, index=0, width=1280, height=720, fps=30, opener=None):
        """opener: optional function returning a capture object (used by tests to simulate failures)."""
        self.index, self.size, self.fps = index, (width, height), fps
        self.opener = opener or self._open
        self.cap = self.opener()
        self.frame, self.frame_id, self.frame_time = None, 0, 0.0
        self.ok = self.cap.isOpened()
        self.connected, self.reconnects = self.ok, 0
        self.cond = threading.Condition()
        self.running = True
        self.camera_fps = 0.0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _open(self):
        # DirectShow first: on Windows the default (Media Foundation) halves the frame rate in dim light (15 fps
        # = 33 ms more delay), DirectShow kept 30 fps with the same auto-exposure on the test webcam
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW) if sys.platform == "win32" else None
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(self.index)
        else:
            # webcams remember a manual exposure inside the device (set by any app, or a test), which left the
            # picture at 15/255 brightness and untrackable: always switch automatic exposure back on
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        if self.fps > 30:  # webcams that do 60 fps at 720p mostly do it only in their compressed (MJPG) mode
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        t_prev, fails = time.perf_counter(), 0
        while self.running:
            try:
                ok, frame = self.cap.read()
            except cv2.error:  # (a driver error counts as a failed read, it doesn't end the reader)
                ok, frame = False, None
            now = time.perf_counter()
            if ok:
                fails = 0
                with self.cond:
                    self.frame, self.frame_id, self.frame_time = frame, self.frame_id + 1, now
                    self.connected = True
                    self.camera_fps = 0.9 * self.camera_fps + 0.1 / max(now - t_prev, 1e-6)
                    t_prev = now
                    self.cond.notify_all()
                continue
            fails += 1
            if fails >= 5:  # lost: release and keep trying to reopen, once a second
                with self.cond:
                    self.connected = False
                    self.cond.notify_all()
                self.cap.release()
                time.sleep(1.0)
                if not self.running:
                    break
                try:
                    self.cap = self.opener()
                except Exception as e:  # keep trying: the reader thread must never die
                    print(f"camera reopen failed ({e}); retrying", flush=True)
                    continue
                self.reconnects += 1
                fails = 0
            else:
                time.sleep(0.02)

    def read(self, last_id=0, timeout=0.5):
        """Newest frame that is newer than last_id: (ok, frame, frame_id). ok is False while waiting for a
        reconnect; the caller keeps running (the avatar eases to rest) instead of quitting."""
        with self.cond:
            self.cond.wait_for(lambda: self.frame_id > last_id, timeout)
            if self.frame is None or self.frame_id <= last_id:
                return False, None, last_id
            return True, self.frame, self.frame_id

    def release(self):
        """Stop the reader first, then close the camera (closing it during a read can crash the driver)."""
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()
