"""VTube Studio (Live2D) output through its public plugin API (ws://localhost:8001).

The first time it connects, VTube Studio shows a popup asking to allow "Avatar Studio Tracker"; click Allow.
The token is saved in vts_token.json so later starts connect without asking. Face values are injected into
VTube Studio's standard tracking parameters (FaceAngleX, MouthOpen, EyeOpenLeft, ...), so any Live2D model set
up for normal webcam/iPhone tracking reacts to them. Turn off VTube Studio's own camera tracking.
"""
import json
import os
import threading
import time
from avatarstudio.core.paths import user_file

TOKEN_PATH = user_file("vts_token.json")
PLUGIN = {"pluginName": "Avatar Studio Tracker", "pluginDeveloper": "Avatar Studio"}


def vts_parameters(res, avatar, calib):
    """Map tracking + avatar output to VTube Studio's default input parameters."""
    face = max(res["faces"], key=lambda fc: fc["box"][2]) if res["faces"] else None
    if face is None:
        return None
    clip = lambda v, lo, hi: max(lo, min(hi, float(v)))
    yaw, pitch, roll = (face["head_pose"][k] for k in ("yaw", "pitch", "roll"))
    if calib and "head_angles" in calib:
        n = calib["head_angles"]
        yaw, pitch, roll = yaw - n[0], pitch - n[1], roll - n[2]
    a, v = avatar["arkit"], avatar["vrm"]
    g = lambda k: a.get(k, 0.0)
    gx, gy = avatar.get("gaze") or (0.0, 0.0)  # calibrated when the G calibration has been done
    gaze = {"x": gx, "y": gy}
    root = avatar["root"]
    mouth_open = clip(max(sum(v.get(x, 0.0) for x in "AIUEO"), g("jawOpen")), 0, 1)
    params = {
        "FaceAngleX": clip(yaw, -30, 30), "FaceAngleY": clip(pitch, -30, 30), "FaceAngleZ": clip(-roll, -90, 90),
        "FacePositionX": clip(root[0] * 100, -15, 15), "FacePositionZ": clip(root[2] * 100, -10, 10),
        "MouthOpen": mouth_open,
        "MouthSmile": clip((g("mouthSmileLeft") + g("mouthSmileRight")) / 2, 0, 1),
        "MouthX": clip(g("mouthLeft") - g("mouthRight"), -1, 1),
        "EyeOpenLeft": clip(1 - v.get("Blink_L", g("eyeBlinkLeft")), 0, 1),
        "EyeOpenRight": clip(1 - v.get("Blink_R", g("eyeBlinkRight")), 0, 1),
        "EyeLeftX": clip(gaze["x"], -1, 1), "EyeRightX": clip(gaze["x"], -1, 1),
        "EyeLeftY": clip(-gaze["y"], -1, 1), "EyeRightY": clip(-gaze["y"], -1, 1),
        "Brows": clip(g("browInnerUp"), 0, 1),
        "BrowLeftY": clip(0.5 + 0.5 * (g("browOuterUpLeft") - g("browDownLeft")), 0, 1),
        "BrowRightY": clip(0.5 + 0.5 * (g("browOuterUpRight") - g("browDownRight")), 0, 1),
        "CheekPuff": clip(g("cheekPuff"), 0, 1),
        "FaceAngry": clip(v.get("Angry", 0.0), 0, 1),
    }
    return [{"id": k, "value": round(val, 4)} for k, val in params.items()]


class VTubeStudio:
    """Background connection to VTube Studio; send() hands it the latest values, it injects them."""

    def __init__(self, url="ws://127.0.0.1:8001"):
        self.url, self.state, self.latest, self.running = url, "connecting", None, True
        self.lock = threading.Lock()
        self._id = 0
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, params):
        with self.lock:
            self.latest = params

    def stop(self):
        self.running = False

    def _request(self, ws, msg_type, data=None):
        self._id += 1
        ws.send(json.dumps({"apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0", "requestID": str(self._id),
                            "messageType": msg_type, "data": data or {}}))
        return json.loads(ws.recv(timeout=60 if msg_type == "AuthenticationTokenRequest" else 5))

    def _auth(self, ws):
        from avatarstudio.core.paths import load_json
        saved = load_json(TOKEN_PATH, {})
        token = saved.get("token") if isinstance(saved, dict) else None
        for attempt in range(2):
            if not token:
                self.state = "waiting - click Allow in VTube Studio"
                r = self._request(ws, "AuthenticationTokenRequest", PLUGIN)
                token = r.get("data", {}).get("authenticationToken")
                if not token:
                    raise RuntimeError(r.get("data", {}).get("message", "permission denied in VTube Studio"))
                from avatarstudio.core.paths import save_json
                save_json(TOKEN_PATH, {"token": token})
            r = self._request(ws, "AuthenticationRequest", {**PLUGIN, "authenticationToken": token})
            if r.get("data", {}).get("authenticated"):
                return
            token = None  # token revoked in VTube Studio: ask again
        raise RuntimeError("VTube Studio refused authentication")

    def _run(self):
        from websockets.sync.client import connect
        while self.running:
            try:
                with connect(self.url, open_timeout=3) as ws:
                    self._auth(ws)
                    self.state = "connected"
                    last, fresh = None, 0.0
                    while self.running:
                        with self.lock:
                            params, self.latest = self.latest, None
                        now = time.time()
                        if params is not None:
                            fresh = now
                        # keep-alive for short gaps (VTS drops data older than ~1 s); after 2 s without a face, stop
                        # so VTube Studio's own "face lost" behaviour takes over
                        if params is not None or (last and now - last[0] > 0.5 and now - fresh < 2.0):
                            values = params if params is not None else last[1]
                            self._request(ws, "InjectParameterDataRequest",
                                          {"faceFound": True, "mode": "set", "parameterValues": values})
                            last = (now, values)
                        else:
                            time.sleep(0.005)
            except Exception as e:  # not running yet, closed, or refused: retry
                self.state = f"not connected ({type(e).__name__}) - is VTube Studio running with its API on?"
                time.sleep(3)
        self.state = "off"
