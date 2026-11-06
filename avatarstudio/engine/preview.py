"""The studio's picture for the Avatar Studio app window (studio --ui): the camera view with what's tracked drawn on
it, as a JPEG stream at http://127.0.0.1:8000/camera.mjpg instead of a window of its own.

Each viewer gets the newest picture whenever there is one (a slow viewer skips pictures; nothing queues up), and
nothing is drawn or encoded while nobody watches (the app on another page, or showing only the avatar).

ToolView does the same for the tools that had a camera window of their own (face training, gesture recording).
"""
import threading
import time


class PreviewHub:
    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg, self.count, self.viewers = None, 0, 0

    def watched(self):
        return self.viewers > 0

    def publish(self, jpeg):
        with self.cond:
            self.jpeg, self.count = jpeg, self.count + 1
            self.cond.notify_all()

    def serve(self, handler):
        """An HTTP GET of camera.mjpg (on the server's thread for that connection): multipart JPEGs until the
        viewer goes away."""
        handler.send_response(200)
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        with self.cond:
            self.viewers += 1
        seen = -1
        try:
            while True:
                with self.cond:
                    self.cond.wait_for(lambda: self.count != seen, timeout=2.0)
                    jpeg, seen = self.jpeg, self.count
                if jpeg is None:
                    continue
                handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg))
                handler.wfile.write(jpeg)
                handler.wfile.write(b"\r\n")
        except OSError:  # (the viewer closed the connection)
            pass
        finally:
            with self.cond:
                self.viewers -= 1


class ToolView:
    """A tool's camera window, in the Avatar Studio app instead (face_refiner.py / gestures.py with --ui). Tools run
    while tracking is off, so they use the studio's addresses: the picture as camera.mjpg on :8000, their prompt,
    progress (0..1) and status as frame data on ws://127.0.0.1:8765, the app's keys back from there (Q cancels)."""
    WIDTH = 960

    def __init__(self, tool):
        from avatarstudio.engine.streaming import WebSocketStream, serve_viewer
        self.tool, self.hub = tool, PreviewHub()
        self.ws = WebSocketStream()
        serve_viewer(routes={"/camera.mjpg": self.hub.serve})
        self.t_prev, self.fps = time.time(), 0.0

    def show(self, view, prompt=None, progress=None, **status):
        """view: the picture (BGR); prompt: lines for the app to show over its page; status: e.g. face=True."""
        import cv2
        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self.t_prev, 1e-3)
        self.t_prev = now
        if self.hub.watched():
            h, w = view.shape[:2]
            if w > self.WIDTH:
                view = cv2.resize(view, (self.WIDTH, round(h * self.WIDTH / w)), interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.hub.publish(jpeg.tobytes())
        self.ws.send({"type": "frame", "prompt": prompt, "progress": progress,
                      "status": {"tool": self.tool, "fps": self.fps, **status}})

    def keys(self):
        """Keys pressed in the app since the last call (lower case)."""
        return [str(m["key"])[:1].lower() for m in self.ws.messages()
                if isinstance(m, dict) and m.get("type") == "key" and m.get("key")]
