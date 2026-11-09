"""Installs the WiLoR finger helper for you (Settings > WiLoR finger helper in the app, or
`AvatarStudio.exe wilor-setup --mano MANO_RIGHT.pkl`), instead of the manual setup in docs/TECHNICAL.md.

Everything goes into one folder, %LOCALAPPDATA%\\AvatarStudio\\wilor (AVATARSTUDIO_WILOR_DIR overrides it):
  1. Python 3.11 (python.org's embeddable package, checked against its SHA-256) and pip (a pinned wheel, checked)
  2. PyTorch 2.5.1 + torchvision: the CUDA build with an NVIDIA GPU, else the CPU build (~2.5 / ~0.3 GB)
  3. WiLoR-mini at a pinned commit (checked), with its torch pin loosened and the chumpy line removed (as the
     manual setup does), plus dill
  4. your MANO_RIGHT.pkl (from mano.is.tue.mpg.de: their licence needs you to register, so the app can't fetch it),
     rewritten without chumpy objects; only numpy / scipy data is read from it, nothing else in the file runs
  5. a first start of WiLoR, which downloads its weights and hand detector from WiLoR-mini's Hugging Face page
Then READY is written and the studio uses it (wilor_client.find_wilor). Running it again repairs or finishes an
installation; a step that's done is skipped. Removing the folder (or Settings > Uninstall) removes it.

Licences, which you accept yourself before it installs: WiLoR's weights are CC BY-NC-ND 4.0 (non-commercial), MANO is
for non-commercial research, education and artistic projects, the hand detector (ultralytics) is AGPL-3.0.
"""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

from avatarstudio.core.paths import cache_dir

PYTHON = ("https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip",
          "009d6bf7e3b2ddca3d784fa09f90fe54336d5b60f0e0f305c37f400bf83cfd3b")
PIP = ("https://files.pythonhosted.org/packages/d4/55/90db48d85f7689ec6f81c0db0622d704306c5284850383c090e6c7195a5c/"
       "pip-24.2-py3-none-any.whl", "2cd581cf58ab7fcfca4ce8efa6dcacd0de5bf8d0a3eb9ec927e07405f4d9e2a2")
WILOR_MINI = ("https://codeload.github.com/warmshao/WiLoR-mini/zip/ebec42f94c389070cdd7dda6fd1bf0b4a659c960",
              "fb7d6d5b66779a47fb371a376524e0b991e9decbc70429cbe321da1781281949")
TORCH = ["torch==2.5.1", "torchvision==0.20.1"]
TORCH_INDEX = {"cuda": "https://download.pytorch.org/whl/cu124", "cpu": "https://download.pytorch.org/whl/cpu"}
STEPS = 6
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def wilor_dir():
    return os.environ.get("AVATARSTUDIO_WILOR_DIR") or os.path.join(cache_dir(), "wilor")


def installed_python():
    """The installed helper's python.exe when the installation finished, else None."""
    d = wilor_dir()
    py = os.path.join(d, "python", "python.exe")
    return py if os.path.exists(os.path.join(d, "READY")) and os.path.exists(py) else None


def models_dir(d):
    return os.path.join(d, "python", "Lib", "site-packages", "wilor_mini", "pretrained_models")


def state():
    """For the app: {"installed", "mano" (MANO_RIGHT.pkl in place), "folder"}."""
    d = wilor_dir()
    return {"installed": installed_python() is not None, "folder": d,
            "mano": os.path.exists(os.path.join(models_dir(d), "MANO_RIGHT.pkl"))}


def step(n, text):
    print(f"step {n}/{STEPS}: {text}", flush=True)


def fetch(url, sha256, dest):
    """url -> dest, which must have this SHA-256 (else it's removed and the setup stops)."""
    h, done, last = hashlib.sha256(), 0, 0
    req = urllib.request.Request(url, headers={"User-Agent": "AvatarStudio-wilor-setup"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest + ".partial", "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        for block in iter(lambda: r.read(1 << 20), b""):
            f.write(block)
            h.update(block)
            done += len(block)
            if total and time.time() - last > 1:
                last = time.time()
                print(f"  {done / 2 ** 20:.0f} of {total / 2 ** 20:.0f} MB", flush=True)
    if h.hexdigest() != sha256:
        os.remove(dest + ".partial")
        raise SystemExit(f"{os.path.basename(url)} didn't match its checksum (changed or damaged); nothing installed")
    os.replace(dest + ".partial", dest)


def run(cmd, what):
    """A step's command, its output passed on (pip's progress); stops the setup if it fails."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", env={**os.environ, "PYTHONUTF8": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
                         **NO_WINDOW)
    for line in p.stdout:
        line = line.rstrip()
        if line:
            print(f"  {line}", flush=True)
    if p.wait():
        raise SystemExit(f"{what} failed (exit code {p.returncode}); run the setup again to retry")


def nvidia():
    """An NVIDIA GPU with its driver (its nvidia-smi is there): the CUDA build of PyTorch, else the CPU one."""
    system = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
    return os.path.exists(system) or shutil.which("nvidia-smi") is not None


# ---------------------------------------------------------------- MANO_RIGHT.pkl without chumpy (safely)
class _Chumpy:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state


class _SafeUnpickler(pickle.Unpickler):
    """Only plain data: numpy arrays, scipy sparse matrices and chumpy objects (kept as their data). A pickle can
    name any function to call while loading; anything else here is refused instead of run."""
    ALLOWED = ("numpy", "scipy.sparse", "chumpy")

    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Chumpy
        if module in ("copy_reg", "copyreg") and name == "_reconstructor":
            return super().find_class("copyreg", name)
        if module in ("__builtin__", "builtins") and name in ("object", "set", "frozenset", "slice", "complex"):
            return super().find_class("builtins", name)
        if module == "_codecs" and name == "encode":
            return super().find_class(module, name)
        if any(module == m or module.startswith(m + ".") for m in self.ALLOWED):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"{module}.{name} isn't hand-model data")


def _plain(v):
    import numpy as np
    if isinstance(v, _Chumpy):
        st = v.__dict__.get("_state", {})
        if isinstance(st, dict) and "idxs" in st and "a" in st:  # chumpy Select: a.ravel()[idxs] -> shape
            out = np.asarray(_plain(st["a"])).ravel()[np.asarray(st["idxs"])]
            shape = st.get("preferred_shape")
            return out.reshape(shape) if shape is not None else out
        for key in ("x", "r", "a"):
            if isinstance(st, dict) and isinstance(st.get(key), np.ndarray):
                return st[key]
        raise ValueError("an unknown chumpy object in the file")
    return v


def install_mano(src, d):
    """Your MANO_RIGHT.pkl -> WiLoR's models folder, as plain arrays (smplx can't load chumpy on new Pythons)."""
    with open(src, "rb") as f:
        data = _SafeUnpickler(f, encoding="latin1").load()
    if not isinstance(data, dict) or "v_template" not in data or "J_regressor" not in data:
        raise SystemExit("That file isn't the MANO hand model: pick mano_v1_2/models/MANO_RIGHT.pkl from "
                         "mano.is.tue.mpg.de's Models & Code download")
    if data["v_template"].shape != (778, 3):
        raise SystemExit("that's not the right-hand MANO model (MANO_RIGHT.pkl)")
    clean = {k: _plain(v) for k, v in data.items()}
    os.makedirs(models_dir(d), exist_ok=True)
    out = os.path.join(models_dir(d), "MANO_RIGHT.pkl")
    with open(out + ".partial", "wb") as f:
        pickle.dump(clean, f, protocol=4)
    os.replace(out + ".partial", out)


# ---------------------------------------------------------------- the steps
def setup(mano=None):
    if sys.platform != "win32":
        raise SystemExit("The automatic setup is for Windows; elsewhere follow the manual WiLoR setup in docs/TECHNICAL.md")
    d = wilor_dir()
    os.makedirs(d, exist_ok=True)
    py_dir, py = os.path.join(d, "python"), os.path.join(d, "python", "python.exe")
    have_mano = os.path.exists(os.path.join(models_dir(d), "MANO_RIGHT.pkl"))
    if not mano and not have_mano:
        raise SystemExit("Pick your MANO_RIGHT.pkl first (from mano.is.tue.mpg.de, see the licence note)")
    if mano and not os.path.isfile(mano):
        raise SystemExit(f"No file at {mano}")
    try:
        os.remove(os.path.join(d, "READY"))
    except OSError:
        pass
    tmp = tempfile.mkdtemp(prefix="avs-wilor-")
    try:
        step(1, "Python 3.11 for the helper")
        if not os.path.exists(py):
            fetch(PYTHON[0], PYTHON[1], os.path.join(tmp, "python.zip"))
            shutil.rmtree(py_dir + ".partial", ignore_errors=True)
            with zipfile.ZipFile(os.path.join(tmp, "python.zip")) as z:
                z.extractall(py_dir + ".partial")
            # the embeddable Python ignores site-packages until its ._pth file says so; the helper's scripts are
            # run from %TEMP%\AvatarStudio-wilor (wilor_client.py), so that folder is on its path too
            pth = next(f for f in os.listdir(py_dir + ".partial") if f.endswith("._pth"))
            lines = [l for l in open(os.path.join(py_dir + ".partial", pth)).read().splitlines() if l.strip()]
            lines = [l for l in lines if l.strip() != "#import site"]
            lines += ["Lib\\site-packages", os.path.join(tempfile.gettempdir(), "AvatarStudio-wilor"), "import site"]
            open(os.path.join(py_dir + ".partial", pth), "w").write("\n".join(lines) + "\n")
            os.replace(py_dir + ".partial", py_dir)
        else:
            print("  already there", flush=True)

        step(2, "pip")
        if subprocess.run([py, "-m", "pip", "--version"], capture_output=True, **NO_WINDOW).returncode:
            whl = os.path.join(tmp, PIP[0].rsplit("/", 1)[1])
            fetch(PIP[0], PIP[1], whl)
            run([py, os.path.join(whl, "pip"), "install", "--no-warn-script-location", whl], "Installing pip")
            run([py, "-m", "pip", "install", "--no-warn-script-location", "setuptools", "wheel"], "Installing setuptools")
        else:
            print("  already there", flush=True)

        kind = "cuda" if nvidia() else "cpu"
        step(3, f"PyTorch ({'NVIDIA GPU' if kind == 'cuda' else 'no NVIDIA GPU found: CPU build, slower'}); "
                f"the biggest download, {'about 2.5 GB' if kind == 'cuda' else 'about 300 MB'}")
        run([py, "-m", "pip", "install", "--no-warn-script-location", "--progress-bar", "off", *TORCH,
             "--index-url", TORCH_INDEX[kind]], "Installing PyTorch")

        step(4, "WiLoR-mini")
        run([py, "-m", "pip", "install", "--no-warn-script-location", "setuptools", "wheel"], "Installing setuptools")
        fetch(WILOR_MINI[0], WILOR_MINI[1], os.path.join(tmp, "wilor_mini.zip"))
        with zipfile.ZipFile(os.path.join(tmp, "wilor_mini.zip")) as z:
            z.extractall(os.path.join(tmp, "src"))
        src = os.path.join(tmp, "src", os.listdir(os.path.join(tmp, "src"))[0])
        req = os.path.join(src, "requirements.txt")
        lines = [("torch>=2.0" if l.startswith("torch>") or l.startswith("torch<") else l)
                 for l in open(req).read().splitlines() if l.strip() and "chumpy" not in l]
        open(req, "w").write("\n".join(lines) + "\n")
        run([py, "-m", "pip", "install", "--no-warn-script-location", "--progress-bar", "off",
             "--no-build-isolation", src, "dill"], "Installing WiLoR-mini")

        step(5, "your MANO hand model")
        if mano:
            install_mano(mano, d)
            print("  MANO_RIGHT.pkl in place", flush=True)
        else:
            print("  already there", flush=True)

        step(6, "WiLoR's weights and hand detector (first start, downloads about 1 GB once)")
        run([py, "-c", "import torch; from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import "
                       "WiLorHandPose3dEstimationPipeline as P; P(device=torch.device('cpu'), dtype=torch.float32, "
                       "verbose=False); print('WiLoR loads, GPU:', torch.cuda.is_available())"], "Starting WiLoR")
        with open(os.path.join(d, "READY"), "w") as f:
            json.dump({"torch": kind, "python": "3.11.9", "wilor_mini": WILOR_MINI[0].rsplit("/", 1)[1]}, f)
        print("WiLoR finger helper installed: it starts with the next tracking (Settings: WiLoR fingers)", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mano", help="your MANO_RIGHT.pkl (needed once)")
    ap.add_argument("--state", action="store_true", help="print whether it's installed (JSON)")
    args = ap.parse_args()
    if args.state:
        print(json.dumps(state()))
        return
    setup(args.mano)


if __name__ == "__main__":
    main()
