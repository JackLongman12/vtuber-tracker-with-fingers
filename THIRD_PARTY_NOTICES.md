# Third-party notices

Avatar Studio's own code is under the MIT licence ([LICENSE](LICENSE)). The libraries and models below keep their
own licences. Some of them, and some of the data the models were trained on, allow **non-commercial use only**, so
Avatar Studio is a free hobby project for personal, non-commercial use.

Full licence texts: in the app, the `licenses` folder next to it (`%LOCALAPPDATA%\AvatarStudio\app\<version>\licenses`;
`INDEX.txt` lists every package), collected by `tools/build_dist.py` from the installed packages; the models' and
the 3D viewer's are in [`packaging/licenses`](packaging/licenses) and `avatarstudio/ui/web/lib/*/LICENSE`.

## In the app (AvatarStudio.exe)

| Component | Used for | Licence |
|---|---|---|
| Python 3.11 | runs the app | PSF License |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | face, hand and body tracking | Apache-2.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) (DirectML) | the GPU models | MIT |
| [OpenCV](https://github.com/opencv/opencv-python) | camera, image processing | Apache-2.0 |
| NumPy, SciPy, scikit-learn, joblib, threadpoolctl | maths, gestures, mouth model | BSD-3-Clause (NumPy also has 0BSD, MIT, Zlib and CC0 parts) |
| websockets | the connection between the app, the engine and the 3D viewer | BSD-3-Clause |
| sounddevice (with PortAudio) | microphone | MIT |
| [pynput](https://github.com/moses-palmer/pynput) | hotkeys | LGPL-3.0 |
| [pystray](https://github.com/moses-palmer/pystray) | tray icon | LGPL-3.0 |
| Pillow | tray icon image | MIT-CMU |
| [pywebview](https://github.com/r0x0r/pywebview), bottle, proxy_tools | the app window | BSD-3-Clause, MIT, MIT |
| Microsoft Edge WebView2 SDK (the DLLs that come with pywebview) | the app window | Microsoft's WebView2 SDK licence |
| pythonnet, clr-loader | the app window (.NET) | MIT |
| Tcl/Tk | the classic launcher | Tcl/Tk licence (BSD-style) |
| [PyInstaller](https://pyinstaller.org) | its bootloader starts the app | GPL-2.0-or-later with an exception that allows shipping it with any program |
| [three.js](https://github.com/mrdoob/three.js) 0.160.0 | 3D viewer | MIT |
| [three-vrm](https://github.com/pixiv/three-vrm) 3.4.0 | 3D viewer (VRM avatars) | MIT |
| three.js's loaders for glTF / GLB, FBX and MMD, with [fflate](https://github.com/101arrowz/fflate), [mmd-parser](https://github.com/takahirox/mmd-parser) and the [meshoptimizer](https://github.com/zeux/meshoptimizer) decoder inside | 3D viewer (other avatar formats) | MIT |
| [Draco](https://github.com/google/draco) decoder (`draco_decoder.wasm`, `draco_wasm_wrapper.js`, shipped with three.js) | compressed glTF avatars | Apache-2.0 |

Smaller dependencies (cffi, pycparser, protobuf, flatbuffers, absl-py, onnx, sympy, six, typing_extensions, ...) are
under MIT, BSD, Apache-2.0 or PSF licences, certifi and tqdm under MPL-2.0 (included unchanged; their source is on
PyPI); `licenses/INDEX.txt` in the app lists them all.

pynput and pystray (LGPL-3.0) are included unchanged. Avatar Studio's source is public, so you can build the app
with your own versions of them ([docs/DEVELOPMENT.md](docs/DEVELOPMENT.md), "Building"); the LGPL and GPL texts are
in the app's `licenses` folder.

## Models (models/, in the app and in models.zip)

| File | What | From | Licence |
|---|---|---|---|
| `face_landmarker.task`, `hand_landmarker.task`, `pose_landmarker_full.task`, `pose_landmarker_lite.task` | face, hands, body | [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) (Google) | Apache-2.0 |
| `mp_pose_face_palm.onnx`, `mp_hand_landmarks.onnx`, `mp_face_blendshapes.onnx` | the same MediaPipe models converted to ONNX by `tools/convert_landmark_models.py` (changed file format, same weights) | MediaPipe (Google) | Apache-2.0 |
| `rtmw-x-wholebody-384x288.onnx` | RTMW-x whole-body model | [OpenMMLab mmpose](https://github.com/open-mmlab/mmpose) | Apache-2.0 (training data: see below) |
| `silero_vad.onnx` | speech detection | [Silero VAD](https://github.com/snakers4/silero-vad) | MIT |
| `rnnoise.dll` | noise suppression | [RNNoise](https://gitlab.xiph.org/xiph/rnnoise) (Xiph.Org, Mozilla), built in the pyrnnoise package | BSD-3-Clause |
| `backbone.onnx` | image features for the mouth model | torchvision MobileNetV3-Large, ImageNet weights, exported by `tools/export_backbone.py` | BSD-3-Clause |

**Training data.** RTMW was trained on a mix of 14 datasets, including COCO-WholeBody, whose authors allow it
["only for research and non-commercial use"](https://github.com/jin-s13/COCO-WholeBody). The mouth model's backbone
was trained on ImageNet, which is for non-commercial research. The model licences above don't add these limits, but
they are one more reason Avatar Studio is for non-commercial use.

## Optional: WiLoR fingers (not included)

The WiLoR finger helper (`avatarstudio/tracking/wilor_server.py`) runs in a separate environment, `.venv_wilor`,
which you set up yourself (docs/TECHNICAL.md, "WiLoR fingers"). None of the following is in the app, in models.zip or in this
repository; by installing them you accept their licences yourself:

| Component | Licence |
|---|---|
| [WiLoR](https://github.com/rolpotamias/WiLoR) model (Potamias et al., CVPR 2025) | [CC BY-NC-ND 4.0](https://raw.githubusercontent.com/rolpotamias/WiLoR/main/license.txt): **non-commercial**; changed weights may not be shared |
| [MANO](https://mano.is.tue.mpg.de) hand model (Max Planck Institute for Intelligent Systems) | [MANO licence](https://mano.is.tue.mpg.de/license.html): **non-commercial** research, education or artistic projects; register and download it yourself; don't share it |
| [WiLoR-mini](https://github.com/warmshao/WiLoR-mini) | no licence published |
| [ultralytics](https://github.com/ultralytics/ultralytics) (the hand detector WiLoR-mini uses) | AGPL-3.0 |
| [smplx](https://github.com/vchoutas/smplx) | its own non-commercial research licence (MPI) |
| PyTorch, torchvision, timm | BSD-3-Clause, BSD-3-Clause, Apache-2.0 |

`wilor_server.py` itself is Avatar Studio code under MIT (MIT code may be combined with AGPL code).

## Not included

Avatars (VRM files), recordings, calibration and training data are your own and are part of neither the app nor
this repository.
