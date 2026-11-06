"""Rewrite MANO_RIGHT.pkl without chumpy objects (plain numpy arrays) so smplx can load it on modern Python.
Keeps MANO_RIGHT.pkl.chumpy as a backup."""
import os
import pickle
import shutil
import sys

import numpy as np

path = sys.argv[1]


class Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return Stub
        return super().find_class(module, name)


def to_array(v):
    if isinstance(v, Stub):
        st = v.__dict__.get("_state", {})
        if isinstance(st, dict) and "idxs" in st and "a" in st:  # chumpy Select: a.ravel()[idxs] -> shape
            src = np.asarray(to_array(st["a"]))
            out = src.ravel()[np.asarray(st["idxs"])]
            shape = st.get("preferred_shape")
            return out.reshape(shape) if shape is not None else out
        for key in ("x", "r", "a"):
            if isinstance(st, dict) and key in st and isinstance(st[key], np.ndarray):
                return st[key]
        raise ValueError(f"unknown chumpy state keys: {list(st) if isinstance(st, dict) else type(st)}")
    return v


with open(path, "rb") as f:
    data = Unpickler(f, encoding="latin1").load()
clean = {}
for k, v in data.items():
    clean[k] = to_array(v)
    print(f"{k:<20} {type(v).__name__:<10} -> {type(clean[k]).__name__} {getattr(clean[k], 'shape', '')}")
if not os.path.exists(path + ".chumpy"):
    shutil.copy2(path, path + ".chumpy")
with open(path, "wb") as f:
    pickle.dump(clean, f)
print("rewritten", path)
