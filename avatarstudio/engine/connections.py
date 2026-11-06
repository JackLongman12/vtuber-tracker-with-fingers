"""Is a VTuber app there to receive the avatar? (Check page and first-run guide in the app window.)

VMC goes out as UDP to 127.0.0.1:39539: if no program has that port open, the motion goes nowhere and nothing tells
you. VTube Studio's plugin API is a WebSocket server on port 8001. Both are found by asking the OS which program has
the port open (psutil), so it works before tracking starts and without sending anything.
"""
import os

VMC_PORT, VTS_PORT = 39539, 8001
KNOWN = {"vseeface": "VSeeFace", "vnyan": "VNyan", "warudo": "Warudo", "vmagicmirror": "VMagicMirror",
         "vtube studio": "VTube Studio", "vtubestudio": "VTube Studio", "virtualmotioncapture": "VirtualMotionCapture",
         "unity": "Unity", "blender": "Blender", "luppet": "Luppet", "3tene": "3tene"}


def _app_name(pid):
    import psutil
    try:
        name = psutil.Process(pid).name()
    except Exception:
        return None
    base = os.path.splitext(name)[0]
    low = base.lower()
    return next((v for k, v in KNOWN.items() if k in low), base)


def receivers(vmc_port=VMC_PORT, vts_port=VTS_PORT):
    """{"vmc": {"port", "listening", "app"}, "vts": {...}}: which program (if any) has each port open."""
    import psutil
    me = os.getpid()
    found = {"vmc": None, "vts": None}
    try:
        conns = psutil.net_connections(kind="inet")
    except Exception as e:  # (rare: not allowed to list connections)
        return {"error": f"Couldn't check the ports ({e})"}
    for c in conns:
        if not c.laddr or c.pid == me:
            continue
        if c.laddr.port == vmc_port and c.type == 2:  # SOCK_DGRAM: a UDP receiver
            found["vmc"] = found["vmc"] or (c.pid or 0)
        elif c.laddr.port == vts_port and c.type == 1 and c.status == psutil.CONN_LISTEN:
            found["vts"] = found["vts"] or (c.pid or 0)
    out = {}
    for key, port in (("vmc", vmc_port), ("vts", vts_port)):
        pid = found[key]
        out[key] = {"port": port, "listening": pid is not None, "app": _app_name(pid) if pid else None}
    return out
