"""Live settings for the studio: changed from the settings page (http://127.0.0.1:8000/settings.html),
saved in settings.json, and as named profiles (settings + calibration) in profiles/."""
import json
import math
import os
from dataclasses import asdict, dataclass, fields
from avatarstudio.core.paths import load_json, save_json, user_file

SETTINGS_PATH = user_file("settings.json")
PROFILE_DIR = user_file("profiles")
HOTKEYS_PATH = user_file("hotkeys.json")
DEFAULT_HOTKEYS = {
    "_help": "Expression hotkeys: key -> expression clip, toggled on/off. Keys 1-9 work in the studio window; "
             "the ctrl+alt versions work everywhere (while streaming a game, etc.).",
    "1": "Joy", "2": "Angry", "3": "Sorrow", "4": "Surprised", "5": "Fun",
}


@dataclass
class Settings:
    expressiveness: float = 1.0      # face shape gain
    auto_range: bool = True          # stretch face shapes to your own range
    mic_lipsync: bool = True         # microphone helps the mouth
    mic_weight: float = 0.5          # how much the voice decides the vowel while speaking
    speech_detection: bool = True    # the voice only opens the mouth for speech (not typing, fans, music)
    noise_suppression: bool = False  # RNNoise on the voice: helps in loud rooms, blurs the vowel guess a bit
    idle: bool = True                # breathing, auto blinks, recovery when tracking is lost
    predict_ms: float = 0.0          # look-ahead for every bone; it also amplifies tracking noise (jitter)
    predict_body_ms: float = 10.0    # look-ahead for torso, head and arms only: body lag -17%, shake +8% (take.mp4)
    link_blinks: bool = True         # both eyes blink together unless you clearly wink
    body_follows_head: bool = True   # shoulders out of view: the upper body turns a little with the head
    smoothing: float = 1.0           # < 1 smoother, > 1 snappier
    no_clip: bool = True             # hands / arms never pass through the torso or head
    body_size: float = 1.0           # body thickness for no_clip (bulkier avatar -> higher)
    mirror_avatar: bool = False
    root_motion: bool = True
    camera_fov: float = 65.0         # the webcam's horizontal field of view (degrees): sets the absolute distance
    eye_spacing_cm: float = 6.3      # your pupil-to-pupil distance (cm): the ruler for the distance
    max_fps: float = 0.0             # 0 = as fast as the camera
    vts: bool = False                # send to VTube Studio (Live2D)
    wholebody: bool = True           # GPU whole-body model: left/right of crossing hands
    gpu_tracking: bool = False       # body + face models on the GPU (gpu_tracking.py); from the next start
    wilor_fingers: bool = True       # finger shapes from WiLoR (GPU helper, .venv_wilor) when it's installed
    face_refiner: bool = True        # use face_refiner.json (from "Train Face.bat") when it exists
    mouth_model: bool = True         # tongue / cheek puff / funnel model (from "Train Face.bat"), if trained
    show_mesh: bool = True
    show_body: bool = True
    show_face_monitor: bool = True
    show_panel: bool = True
    version: int = 2                 # settings.json format; older files get changed defaults applied once

    @classmethod
    def load(cls, path=SETTINGS_PATH):
        s = cls()
        data = load_json(path, {})
        if not isinstance(data, dict):
            data = {}
        if data and data.get("version", 1) < 2:  # defaults changed after measuring jitter (see docs/TECHNICAL.md "Less jitter")
            data.update(predict_ms=0.0, version=2)
        s.update(data)
        return s

    def save(self, path=SETTINGS_PATH):
        save_json(path, asdict(self), indent=2)

    def update(self, values):
        """Apply {name: value}, converting to each field's type; unknown names and values that don't fit the
        field's type (a typo in settings.json, a bad message) are ignored."""
        types = {f.name: f.type for f in fields(self)}
        changed = []
        for k, v in values.items():
            if k not in types:
                continue
            try:
                if types[k] is bool:
                    v = v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
                elif types[k] in (float, int):
                    v = types[k](v)
                    if not math.isfinite(v):
                        continue
            except (TypeError, ValueError):
                continue
            setattr(self, k, v)
            changed.append(k)
        return changed

    def as_dict(self):
        return asdict(self)


def list_profiles():
    if not os.path.isdir(PROFILE_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(PROFILE_DIR) if f.endswith(".json"))


def save_profile(name, settings, calibration):
    os.makedirs(PROFILE_DIR, exist_ok=True)
    safe = "".join(c for c in name if c.isalnum() or c in " -_").strip() or "profile"
    save_json(os.path.join(PROFILE_DIR, f"{safe}.json"), {"settings": settings.as_dict(), "calibration": calibration},
              indent=2)
    return safe


def load_profile(name):
    safe = "".join(c for c in name if c.isalnum() or c in " -_").strip()  # (names come from the settings page)
    with open(os.path.join(PROFILE_DIR, f"{safe}.json")) as f:
        return json.load(f)


def load_hotkeys():
    if not os.path.exists(HOTKEYS_PATH):
        save_json(HOTKEYS_PATH, DEFAULT_HOTKEYS, indent=2)
    keys = load_json(HOTKEYS_PATH, None)
    if not isinstance(keys, dict):
        keys = DEFAULT_HOTKEYS
    return {str(k): str(v) for k, v in keys.items() if not str(k).startswith("_")}


class GlobalHotkeys:
    """ctrl+alt+<key> toggles work even when another window (game, OBS) is focused. Only the configured
    combinations are listened for. actions: {key: function} for studio actions (Ctrl+Alt+C recalibrate,
    Ctrl+Alt+P pause), which win over an expression on the same key."""

    def __init__(self, hotkeys, on_toggle, actions=None):
        try:
            from pynput import keyboard
        except ImportError:
            self.listener = None
            return
        combos = {f"<ctrl>+<alt>+{k}": (lambda name=name: on_toggle(name)) for k, name in hotkeys.items()}
        combos.update({f"<ctrl>+<alt>+{k}": fn for k, fn in (actions or {}).items()})
        try:
            self.listener = keyboard.GlobalHotKeys(combos)
        except ValueError as e:  # a key name pynput doesn't know, from a hand-edited hotkeys.json
            print(f"global hotkeys off: {e}")
            self.listener = None
            return
        self.listener.daemon = True
        self.listener.start()

    def stop(self):
        if self.listener:
            self.listener.stop()
