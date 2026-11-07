"""Outputs for avatar data: VMC protocol over OSC/UDP (VTuber apps) and a WebSocket JSON stream (3D viewer,
games, your own tools)."""
import asyncio
import json
import queue
import socket
import struct
import threading
import time

import numpy as np

from avatarstudio.avatar.rig import REST


def _osc_str(s):
    b = s.encode() + b"\0"
    return b + b"\0" * (-len(b) % 4)


def osc_message(address, *args):
    """Encode one OSC message (strings, ints, floats). Hand-rolled: python-osc's builder also re-parses
    every message it builds, which costs several ms per frame for ~200 VMC messages."""
    tags, data = ",", []
    for a in args:
        if isinstance(a, str):
            tags += "s"
            data.append(_osc_str(a))
        elif isinstance(a, int):
            tags += "i"
            data.append(struct.pack(">i", a))
        else:
            tags += "f"
            data.append(struct.pack(">f", a))
    return _osc_str(address) + _osc_str(tags) + b"".join(data)


def osc_bundle(messages):
    head = b"#bundle\0" + struct.pack(">Q", 1)  # time tag 1 = immediately
    return head + b"".join(struct.pack(">i", len(m)) + m for m in messages)


class VMCSender:
    """VMC protocol 'performer' -> 'marionette' (VSeeFace, VNyan, Warudo, VMagicMirror, ... default port 39539).
    Sends local bone rotations, root position and blendshapes (VRM presets + ARKit names for Perfect Sync)."""

    def __init__(self, host="127.0.0.1", port=39539):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.start = time.time()

    _msg = staticmethod(osc_message)

    def _send(self, msgs):
        # several bundles keep each UDP packet small enough for every receiver
        for i in range(0, len(msgs), 40):
            self.sock.sendto(osc_bundle(msgs[i:i + 40]), self.addr)

    def send(self, avatar):
        msgs = [self._msg("/VMC/Ext/OK", 1), self._msg("/VMC/Ext/T", float(time.time() - self.start))]
        rx, ry, rz = (float(v) for v in avatar["root"])
        msgs.append(self._msg("/VMC/Ext/Root/Pos", "root", rx, ry, rz, 0.0, 0.0, 0.0, 1.0))
        for name, q in avatar["bones"].items():
            px, py, pz = REST[name][1] if name in REST else (0.0, 0.0, 0.0)
            msgs.append(self._msg("/VMC/Ext/Bone/Pos", name, float(px), float(py), float(pz),
                                  float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        for name, v in avatar["vrm"].items():
            msgs.append(self._msg("/VMC/Ext/Blend/Val", name, float(v)))
        for name, v in avatar["arkit"].items():
            # Perfect Sync models name their clips either like ARKit (eyeBlinkLeft) or capitalised (EyeBlinkLeft)
            msgs.append(self._msg("/VMC/Ext/Blend/Val", name, float(v)))
            msgs.append(self._msg("/VMC/Ext/Blend/Val", name[0].upper() + name[1:], float(v)))
        msgs.append(self._msg("/VMC/Ext/Blend/Apply"))
        self._send(msgs)


class WebSocketStream:
    """Broadcasts JSON frames to every connected client (ws://host:port). Runs in a background thread."""

    def __init__(self, host="127.0.0.1", port=8765):
        self.host, self.port = host, port
        self.clients = set()
        self.inbox = queue.Queue()  # messages from the settings page (settings changes, button actions)
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()
        self.ready.wait(5)

    def _run(self):
        from websockets.asyncio.server import serve

        async def handler(ws):
            self.clients.add(ws)
            try:
                await ws.send(json.dumps({"type": "rest", "bones": {k: [p, list(o)] for k, (p, o) in REST.items()}}))
                async for text in ws:
                    try:
                        self.inbox.put(json.loads(text))
                    except ValueError:
                        pass
            except Exception:
                pass
            finally:
                self.clients.discard(ws)

        def check(connection, request):
            """Only Avatar Studio's own pages and local programs (core/localweb.py): any website open in a browser
            could otherwise read the tracking data and send settings."""
            from avatarstudio.core.localweb import allowed
            if not allowed(request.headers):
                return connection.respond(403, "Only Avatar Studio's own pages can use this address\n")
            return None

        async def main():
            async with serve(handler, self.host, self.port, process_request=check, max_size=1 << 20):
                self.ready.set()
                await asyncio.Future()

        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(main())
        except OSError as e:  # port taken (another studio running?): tracking and VMC carry on without the viewer
            print(f"3D viewer stream unavailable: port {self.port} is busy ({e.strerror or e}). "
                  f"Is another Avatar Studio running? (--ws-port picks another port)", flush=True)
            self.ready.set()

    def messages(self):
        """Drain messages received from pages since the last call."""
        out = []
        while not self.inbox.empty():
            out.append(self.inbox.get_nowait())
        return out

    def send(self, payload):
        if not self.clients:
            return
        from websockets.asyncio.server import broadcast
        text = json.dumps(payload, default=_jsonable)
        self.loop.call_soon_threadsafe(broadcast, set(self.clients), text)


def serve_viewer(port=8000, directory=None, routes=None):
    """Serve the 3D viewer page at http://127.0.0.1:port (browsers block WebSockets from file:// pages
    in some setups, so the page is served over http). The same server as the app's own viewer windows use;
    routes: extra paths answered by functions (the camera picture for the app, preview.py)."""
    from avatarstudio.core.paths import web_dir
    from avatarstudio.ui.webwindow import serve
    return serve(directory or web_dir(), port, routes)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return np.round(o, 4).tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))
