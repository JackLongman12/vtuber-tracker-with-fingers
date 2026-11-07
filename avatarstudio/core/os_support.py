"""What differs between Windows, macOS and Linux: which GPU runtime ONNX Runtime uses, opening a folder, where a
virtual environment keeps its Python. (Camera, microphone, data folders, tray and the viewer windows handle their
OS differences where they are: camera.py / launcher.py, paths.py, webwindow.py.)

GPU: the first of this OS's runtimes that is installed runs the GPU models (GPU tracking, the whole-body model),
else they run on the CPU. Each comes with its own ONNX Runtime package (requirements.txt picks one per OS):
  Windows  DirectML: any DirectX 12 GPU                                 onnxruntime-directml
  macOS    Core ML: the Metal GPU or the Neural Engine (Core ML picks)  onnxruntime (included)
  Linux    CUDA (NVIDIA), ROCm / MIGraphX (AMD), OpenVINO (Intel)       onnxruntime-gpu / -rocm / -openvino
Only Windows has been measured so far. AVATARSTUDIO_COREML_UNITS=CPUAndGPU keeps Core ML on the GPU (Metal) only,
CPUAndNeuralEngine on the Neural Engine (default ALL), for comparing them on a Mac.
"""
import os
import subprocess
import sys

WINDOWS, MAC, LINUX = sys.platform == "win32", sys.platform == "darwin", sys.platform.startswith("linux")

GPU_PROVIDERS = {"win32": ["DmlExecutionProvider"],
                 "darwin": ["CoreMLExecutionProvider"],
                 "linux": ["CUDAExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider",
                           "OpenVINOExecutionProvider"]}
PROVIDER_OPTIONS = {  # MLProgram: Core ML's newer format, which covers more of the models' operations
    "CoreMLExecutionProvider": {"ModelFormat": "MLProgram",
                                "MLComputeUnits": os.environ.get("AVATARSTUDIO_COREML_UNITS", "ALL")},
    "OpenVINOExecutionProvider": {"device_type": "GPU"}}


def gpu_providers():
    """This OS's GPU runtimes that ONNX Runtime has here, in order of preference."""
    import onnxruntime as ort
    have = ort.get_available_providers()
    return [p for p in GPU_PROVIDERS.get("linux" if LINUX else sys.platform, []) if p in have]


def onnx_session(path, so=None):
    """An ONNX Runtime session on the GPU if this OS's GPU runtime works here, else on the CPU.
    -> (session, the GPU runtime's name or None)."""
    import onnxruntime as ort
    for name in gpu_providers():
        options = PROVIDER_OPTIONS.get(name, {})
        # (an older ONNX Runtime may not know Core ML's options: then its defaults. OpenVINO without its options
        # would run on the CPU, so it gets no second try)
        for options in (options, {}) if name == "CoreMLExecutionProvider" else (options,):
            try:
                sess = ort.InferenceSession(path, so, providers=[(name, options), "CPUExecutionProvider"])
            except Exception:
                continue
            if sess.get_providers()[0] == name:  # (a runtime that can't start falls back to the CPU by itself)
                return sess, name
            break
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]), None


def open_folder(path):
    """Show a folder in Explorer / Finder / the file manager (folders only: os.startfile would run a program)."""
    if not os.path.isdir(path):
        raise NotADirectoryError(path)
    if WINDOWS:
        os.startfile(path)
    else:
        subprocess.Popen(["open" if MAC else "xdg-open", path])


def venv_python(venv):
    """The Python of a virtual environment folder (Windows: Scripts\\python.exe, else bin/python)."""
    return os.path.join(venv, "Scripts", "python.exe") if WINDOWS else os.path.join(venv, "bin", "python")
