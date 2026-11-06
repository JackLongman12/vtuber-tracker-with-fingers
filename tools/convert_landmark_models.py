"""Convert MediaPipe's face, hand and body landmark models (inside models/*.task) to ONNX for gpu_tracking.py.

Developer tool, run once: .venv\\Scripts\\python.exe tools/convert_landmark_models.py [--check]  -> models/mp_*.onnx
The models are copied operator by operator (a small converter for just the operators these four models use;
tflite2onnx fails on three of them), with the same weights (stored as float16 like in the .tflite files, computed
in float32 like MediaPipe does). What gpu_tracking.py loads:
  mp_pose_face_palm.onnx    body + face landmark models and the palm detector side by side: one GPU run per frame
                            (separate runs were waited for about twice as long while WiLoR shares the GPU). Inputs
                            "pose", "face": 1x256x256x3 and "palm": 1x192x192x3 uint8 RGB. The body model's
                            segmentation mask is left out.
  mp_hand_landmarks.onnx    input "image": 1x224x224x3 uint8 RGB crop (runs on the CPU, both hands at once)
  mp_face_blendshapes.onnx  input "points": 1x146x2 face points in pixels
--check compares every output with the tflite interpreter (pip install ai-edge-litert, dev only)."""
import argparse
import os
import zipfile

import numpy as np
import onnx
import tflite
from onnx import TensorProto, helper, numpy_helper

SOURCES = {  # part -> (task file, model inside it)
    "pose": ("models/pose_landmarker_full.task", "pose_landmarks_detector.tflite"),
    "face": ("models/face_landmarker.task", "face_landmarks_detector.tflite"),
    "palm": ("models/hand_landmarker.task", "hand_detector.tflite"),
    "hand": ("models/hand_landmarker.task", "hand_landmarks_detector.tflite"),
    "blendshapes": ("models/face_landmarker.task", "face_blendshapes.tflite"),
}
OUTPUTS = {  # part -> {tflite output: name in the ONNX file} (outputs not listed are left out)
    "pose": {"Identity": "landmarks", "Identity_1": "presence", "Identity_3": "heatmap", "Identity_4": "world"},
    "face": {"Identity": "landmarks", "Identity_1": "presence_logit"},  # (Identity_2: tongue out, unused)
    "palm": {"Identity": "boxes", "Identity_1": "scores"},
    "hand": {"Identity": "landmarks", "Identity_1": "presence", "Identity_2": "handedness", "Identity_3": "world"},
    "blendshapes": {"StatefulPartitionedCall:0": "blendshapes"},
}
FILES = {  # file -> {part: input name}; several parts = one graph with the parts side by side
    "mp_pose_face_palm.onnx": {"pose": "pose", "face": "face", "palm": "palm"},
    "mp_hand_landmarks.onnx": {"hand": "image"},
    "mp_face_blendshapes.onnx": {"blendshapes": "points"},
}
NCHW_OF_NHWC = [0, 2, 3, 1]  # NHWC axis -> its position in NCHW
OPS = {v: k for k, v in tflite.BuiltinOperator.__dict__.items() if not k.startswith("_")}
DTYPES = {tflite.TensorType.FLOAT32: np.float32, tflite.TensorType.FLOAT16: np.float16,
          tflite.TensorType.INT32: np.int32, tflite.TensorType.INT64: np.int64}


class Converter:
    """One tflite model -> ONNX nodes. 4D activations are kept NCHW where an operator prefers it (convolutions,
    pooling, resizing) and transposed back only where a layout-dependent operator needs TF's NHWC order."""

    def __init__(self, buf, prefix=""):
        self.m = tflite.Model.GetRootAsModel(buf, 0)
        self.g = self.m.Subgraphs(0)
        self.prefix = prefix
        self.nodes, self.inits, self.consts, self.vals, self.cache, self.n = [], [], {}, {}, {}, 0

    # ------------------------------------------------------------ helpers
    def shape(self, t):
        return [int(d) for d in self.g.Tensors(t).ShapeAsNumpy()] if self.g.Tensors(t).ShapeLength() else []

    def name(self, t):
        return self.g.Tensors(t).Name().decode()

    def uid(self, base):
        self.n += 1
        return f"{self.prefix}{base}_{self.n}"

    def const(self, arr, base="c"):
        """An initializer; float16 weights stay float16 in the file and are cast to float32 (folded at load)."""
        name = self.uid(base)
        self.inits.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))
        if arr.dtype == np.float16:
            return self.node("Cast", [name], to=TensorProto.FLOAT)
        return name

    def node(self, op, ins, outs=None, **attrs):
        outs = outs or [self.uid(op)]
        self.nodes.append(helper.make_node(op, ins, outs, name=self.uid(op + "_n"), **attrs))
        return outs[0]

    def is4d(self, t):
        return len(self.shape(t)) == 4

    def get(self, t, layout):
        """ONNX value of tflite tensor t in `layout` ('nchw' or 'nhwc'; only matters for 4D tensors)."""
        if t in self.consts:
            a = self.consts[t]
            if layout == "nchw" and a.ndim <= 4:  # (a broadcast constant: to rank 4, then NCHW)
                a = a.reshape((1,) * (4 - a.ndim) + a.shape).transpose(0, 3, 1, 2)
            return self.const(a, "k")
        name, have = self.vals[t]
        if have == layout or not self.is4d(t):
            return name
        key = (t, layout)
        if key not in self.cache:
            self.cache[key] = self.node("Transpose", [name], perm=[0, 3, 1, 2] if layout == "nchw" else [0, 2, 3, 1])
        return self.cache[key]

    def put(self, t, name, layout):
        self.vals[t] = (name, layout if self.is4d(t) else "nhwc")

    def act(self, name, fn):
        A = tflite.ActivationFunctionType
        if fn == A.NONE:
            return name
        if fn == A.RELU:
            return self.node("Relu", [name])
        if fn in (A.RELU6, A.RELU_N1_TO_1):
            lo, hi = (0, 6) if fn == A.RELU6 else (-1, 1)
            return self.node("Clip", [name, self.const(np.float32(lo)), self.const(np.float32(hi))])
        if fn == A.TANH:
            return self.node("Tanh", [name])
        raise NotImplementedError(f"activation {fn}")

    @staticmethod
    def opts(op, cls):
        o = cls()
        t = op.BuiltinOptions()
        o.Init(t.Bytes, t.Pos)
        return o

    @staticmethod
    def same_pads(size, k, stride, dil):
        out = -(-size // stride)
        total = max((out - 1) * stride + (k - 1) * dil + 1 - size, 0)
        return total // 2, total - total // 2

    def pads(self, padding, in_shape, kh, kw, sh, sw, dh=1, dw=1):
        if padding == tflite.Padding.VALID:
            return [0, 0, 0, 0]
        t, b = self.same_pads(in_shape[1], kh, sh, dh)
        l, r = self.same_pads(in_shape[2], kw, sw, dw)
        return [t, l, b, r]

    # ------------------------------------------------------------ conversion
    def run(self, input_name, outputs):
        """-> (nodes, initializers, graph inputs, graph outputs). Images come in as NHWC uint8 RGB (scaled to
        0..1 like MediaPipe's ImageToTensor), other inputs as float32."""
        g, m = self.g, self.m
        for i in range(g.TensorsLength()):  # constants
            tt = g.Tensors(i)
            data = m.Buffers(tt.Buffer()).DataAsNumpy() if tt.Buffer() < m.BuffersLength() else 0
            if not isinstance(data, int) and len(data):
                self.consts[i] = np.frombuffer(data.tobytes(), DTYPES[tt.Type()]).reshape(self.shape(i))
        assert g.InputsLength() == 1
        t = g.Inputs(0)
        s = self.shape(t)
        if len(s) == 4:
            ins = [helper.make_tensor_value_info(input_name, TensorProto.UINT8, s)]
            x = self.node("Cast", [input_name], to=TensorProto.FLOAT)
            self.put(t, self.node("Mul", [x, self.const(np.float32(1 / 255))]), "nhwc")
        else:
            ins = [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, s)]
            self.put(t, input_name, "nhwc")
        for i in range(g.OperatorsLength()):
            self.convert_op(g.Operators(i))
        outs = []
        for k in range(g.OutputsLength()):
            t = g.Outputs(k)
            if self.name(t) in outputs:
                name = self.prefix + outputs[self.name(t)]
                outs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, self.shape(t)))
                self.node("Identity", [self.get(t, "nhwc")], [name])
        return self.nodes, self.inits, ins, outs

    def convert_op(self, op):
        code = self.m.OperatorCodes(op.OpcodeIndex())
        kind = OPS[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())]
        ins = [int(t) for t in op.InputsAsNumpy()] if op.InputsLength() else []
        out = int(op.OutputsAsNumpy()[0])
        T = tflite
        if kind == "DEQUANTIZE":  # float16 weights: kept as they are (cast when emitted)
            self.consts[out] = self.consts[ins[0]]
            return
        if kind in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            o = self.opts(op, T.Conv2DOptions if kind == "CONV_2D" else T.DepthwiseConv2DOptions)
            w = self.consts[ins[1]]
            x_shape = self.shape(ins[0])
            if kind == "CONV_2D":
                wk, group = w.transpose(0, 3, 1, 2), 1
            else:
                wk, group = w.transpose(3, 0, 1, 2), x_shape[3]
            kh, kw = w.shape[1], w.shape[2]
            args = [self.get(ins[0], "nchw"), self.const(wk, "w")]
            if len(ins) > 2 and ins[2] >= 0:
                args.append(self.const(self.consts[ins[2]], "b"))
            y = self.node("Conv", args, kernel_shape=[kh, kw], strides=[o.StrideH(), o.StrideW()],
                          dilations=[o.DilationHFactor(), o.DilationWFactor()], group=group,
                          pads=self.pads(o.Padding(), x_shape, kh, kw, o.StrideH(), o.StrideW(),
                                         o.DilationHFactor(), o.DilationWFactor()))
            self.put(out, self.act(y, o.FusedActivationFunction()), "nchw")
        elif kind == "MAX_POOL_2D":
            o = self.opts(op, T.Pool2DOptions)
            kh, kw = o.FilterHeight(), o.FilterWidth()
            y = self.node("MaxPool", [self.get(ins[0], "nchw")], kernel_shape=[kh, kw],
                          strides=[o.StrideH(), o.StrideW()],
                          pads=self.pads(o.Padding(), self.shape(ins[0]), kh, kw, o.StrideH(), o.StrideW()))
            self.put(out, self.act(y, o.FusedActivationFunction()), "nchw")
        elif kind == "RESIZE_BILINEAR":
            o = self.opts(op, T.ResizeBilinearOptions)
            mode = "align_corners" if o.AlignCorners() else "half_pixel" if o.HalfPixelCenters() else "asymmetric"
            s = self.shape(out)
            y = self.node("Resize", [self.get(ins[0], "nchw"), "", "", self.const(np.int64([s[0], s[3], s[1], s[2]]))],
                          mode="linear", coordinate_transformation_mode=mode)
            self.put(out, y, "nchw")
        elif kind == "PAD":
            p = self.consts[ins[1]].astype(np.int64)
            layout = "nchw" if self.is4d(ins[0]) else "nhwc"
            if layout == "nchw":
                p = p[[0, 3, 1, 2]]
            y = self.node("Pad", [self.get(ins[0], layout), self.const(np.concatenate([p[:, 0], p[:, 1]]))])
            self.put(out, y, layout)
        elif kind in ("ADD", "SUB", "MUL", "DIV", "SQUARED_DIFFERENCE", "PRELU"):
            acts = [t for t in ins if t not in self.consts]
            layout = "nchw" if acts and all(self.is4d(t) for t in acts) and any(
                self.vals[t][1] == "nchw" for t in acts) else "nhwc"
            a, b = self.get(ins[0], layout), self.get(ins[1], layout)
            fn = T.ActivationFunctionType.NONE
            if kind == "SQUARED_DIFFERENCE":
                d = self.node("Sub", [a, b])
                y = self.node("Mul", [d, d])
            elif kind == "PRELU":
                y = self.node("PRelu", [a, b])
            else:
                o = self.opts(op, {"ADD": T.AddOptions, "SUB": T.SubOptions, "MUL": T.MulOptions,
                                   "DIV": T.DivOptions}[kind])
                y = self.node(kind.capitalize(), [a, b])
                fn = o.FusedActivationFunction()
            self.put(out, self.act(y, fn), layout)
        elif kind in ("LOGISTIC", "SQRT", "RSQRT", "NEG"):
            layout = self.vals[ins[0]][1]
            x = self.get(ins[0], layout)
            if kind == "RSQRT":
                y = self.node("Reciprocal", [self.node("Sqrt", [x])])
            else:
                y = self.node({"LOGISTIC": "Sigmoid", "SQRT": "Sqrt", "NEG": "Neg"}[kind], [x])
            self.put(out, y, layout)
        elif kind in ("MEAN", "SUM"):
            keep = self.opts(op, T.ReducerOptions).KeepDims()
            axes = [int(a) % len(self.shape(ins[0])) for a in np.atleast_1d(self.consts[ins[1]])]
            layout = "nhwc"
            if self.is4d(ins[0]) and self.vals[ins[0]][1] == "nchw":
                nhwc_at = [0, 3, 1, 2]  # NCHW position -> NHWC axis
                if keep or [a for a in range(4) if a not in axes] == [a for a in nhwc_at if a not in axes]:
                    layout = "nchw"
            x = self.get(ins[0], layout)
            ax = [NCHW_OF_NHWC[a] for a in axes] if layout == "nchw" else axes
            if kind == "MEAN":
                y = self.node("ReduceMean", [x], axes=ax, keepdims=int(keep))
            else:
                y = self.node("ReduceSum", [x, self.const(np.int64(ax))], keepdims=int(keep))
            self.put(out, y, layout if keep else "nhwc")
        elif kind == "RESHAPE":
            y = self.node("Reshape", [self.get(ins[0], "nhwc"), self.const(np.int64(self.shape(out)))])
            self.put(out, y, "nhwc")
        elif kind == "TRANSPOSE":
            y = self.node("Transpose", [self.get(ins[0], "nhwc")], perm=[int(p) for p in self.consts[ins[1]]])
            self.put(out, y, "nhwc")
        elif kind == "CONCATENATION":
            o = self.opts(op, T.ConcatenationOptions)
            rank = len(self.shape(out))
            axis = o.Axis() % rank
            acts = [t for t in ins if t not in self.consts]
            layout = "nchw" if rank == 4 and any(self.vals[t][1] == "nchw" for t in acts) else "nhwc"
            y = self.node("Concat", [self.get(t, layout) for t in ins],
                          axis=NCHW_OF_NHWC[axis] if layout == "nchw" else axis)
            self.put(out, self.act(y, o.FusedActivationFunction()), layout)
        elif kind == "STRIDED_SLICE":
            o = self.opts(op, T.StridedSliceOptions)
            assert not o.EllipsisMask() and not o.NewAxisMask(), "strided slice masks"
            begin, end, steps = (self.consts[t].astype(np.int64).copy() for t in ins[1:4])
            big = np.iinfo(np.int64).max
            shrink = []
            for a in range(len(begin)):
                if o.BeginMask() >> a & 1:
                    begin[a] = 0 if steps[a] > 0 else big
                if o.EndMask() >> a & 1:
                    end[a] = big if steps[a] > 0 else -big
                if o.ShrinkAxisMask() >> a & 1:
                    end[a], steps[a] = begin[a] + 1, 1
                    shrink.append(a)
            y = self.node("Slice", [self.get(ins[0], "nhwc"), self.const(begin), self.const(end),
                                    self.const(np.arange(len(begin), dtype=np.int64)), self.const(steps)])
            if shrink:
                y = self.node("Squeeze", [y, self.const(np.int64(shrink))])
            self.put(out, y, "nhwc")
        elif kind == "FULLY_CONNECTED":
            o = self.opts(op, T.FullyConnectedOptions)
            w = self.consts[ins[1]]
            x = self.node("Reshape", [self.get(ins[0], "nhwc"), self.const(np.int64([-1, w.shape[1]]))])
            args = [x, self.const(w, "w")]
            if len(ins) > 2 and ins[2] >= 0:
                args.append(self.const(self.consts[ins[2]], "b"))
            y = self.act(self.node("Gemm", args, transB=1), o.FusedActivationFunction())
            y = self.node("Reshape", [y, self.const(np.int64(self.shape(out)))])
            self.put(out, y, "nhwc")
        else:
            raise NotImplementedError(kind)


def read_tflite(part):
    task, inner = SOURCES[part]
    with zipfile.ZipFile(task) as z:
        return z.read(inner)


def prune(nodes, outputs):
    """Drop the nodes no output depends on (e.g. the body model's segmentation head)."""
    need, keep = {o.name for o in outputs}, []
    for n in reversed(nodes):
        if any(o in need for o in n.output):
            keep.append(n)
            need.update(i for i in n.input if i)
    return keep[::-1], need


def build(parts):
    """One ONNX model holding the given parts ({part: input name}) side by side."""
    nodes, inits, ins, outs = [], [], [], []
    for part, input_name in parts.items():
        prefix = f"{part}/" if len(parts) > 1 else ""
        n, i, gi, go = Converter(read_tflite(part), prefix).run(input_name, OUTPUTS[part])
        nodes += n
        inits += i
        ins += gi
        outs += go
    nodes, used = prune(nodes, outs)
    inits = [i for i in inits if i.name in used]
    model = helper.make_model(helper.make_graph(nodes, "mediapipe", ins, outs, inits),
                              opset_imports=[helper.make_opsetid("", 13)], producer_name="avatar-studio")
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def check(path, parts, n=4):
    """Largest difference between the tflite interpreter and the ONNX file (ORT on the CPU)."""
    import onnxruntime as ort
    from ai_edge_litert.interpreter import Interpreter
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = {}
    for k in range(n):
        feed, refs = {}, {}
        for part, input_name in parts.items():
            it = Interpreter(model_content=read_tflite(part))
            it.allocate_tensors()
            inp = it.get_input_details()[0]
            shape = inp["shape"]
            if len(shape) == 4:
                img = (rng.integers(0, 256, shape) if k % 2 else blobs(shape, rng)).astype(np.uint8)
                feed[input_name] = img
                it.set_tensor(inp["index"], img.astype(np.float32) * np.float32(1 / 255))
            else:
                feed[input_name] = (rng.random(shape) * 600).astype(np.float32)
                it.set_tensor(inp["index"], feed[input_name])
            it.invoke()
            prefix = f"{part}/" if len(parts) > 1 else ""
            for o in it.get_output_details():
                if o["name"] in OUTPUTS[part]:
                    refs[prefix + OUTPUTS[part][o["name"]]] = it.get_tensor(o["index"])
        got = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feed)))
        for name, ref in refs.items():
            d = float(np.abs(ref - got[name]).max())
            worst[name] = max(worst.get(name, (0, 0))[0], d), max(worst.get(name, (0, 0))[1], float(np.abs(ref).max()))
    print("   " + ", ".join(f"{k}: max diff {d:.1e} (values up to {s:.3g})" for k, (d, s) in worst.items()))


def blobs(shape, rng):
    """A smooth image-like test input (random blobs) rather than pure noise."""
    _, h, w, c = shape
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    img = np.zeros((h, w, c), np.float32)
    for _ in range(12):
        cx, cy, r = rng.random(3)
        img += rng.random(c).astype(np.float32) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (0.02 + 0.1 * r))[..., None]
    return (img / img.max() * 255)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with the tflite interpreter (needs ai-edge-litert)")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    for file, parts in FILES.items():
        path = os.path.join(args.out, file)
        model = build(parts)
        onnx.save(model, path)
        print(f"{path}: {os.path.getsize(path) / 2 ** 20:.1f} MB, {len(model.graph.node)} nodes")
        if args.check:
            check(path, parts)


if __name__ == "__main__":
    main()
