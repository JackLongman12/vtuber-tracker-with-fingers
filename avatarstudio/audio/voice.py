"""Cleaning up the microphone signal before lip sync.

SpeechDetector  Silero VAD (MIT license, 2 MB ONNX): "is this a human voice?" every 32 ms. Lip sync only
                opens the mouth for speech, so typing, the fan, game audio or a cough don't move it.
Denoiser        RNNoise (BSD license), the same family of noise suppression as Discord's original one: removes
                steady noise (fans, hum, traffic) from the voice. Works at 48 kHz (other rates are converted);
                adds ~10 ms.
VoiceProcessor  raw microphone audio in -> (optionally denoised) 16 kHz voice + speech probability out.
"""
import ctypes
import os
import sys
import threading
import time

import numpy as np

RATE = 16000
VAD_PATH = "models/silero_vad.onnx"
# RNNoise as a library for this OS (the pyrnnoise wheel has each one; only the Windows one is in models/ so far)
RNNOISE_PATH = {"win32": "models/rnnoise.dll", "darwin": "models/librnnoise.dylib"}.get(sys.platform,
                                                                                         "models/librnnoise.so")


class SpeechDetector:
    CHUNK, CONTEXT = 512, 64  # 32 ms at 16 kHz (the model's fixed size) + the end of the previous chunk

    def __init__(self, path=VAD_PATH):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros(self.CONTEXT, np.float32)
        self.sr = np.array(RATE, np.int64)

    def __call__(self, chunk):
        """chunk: 512 float samples at 16 kHz -> probability of speech (0..1)."""
        x = np.concatenate([self.context, chunk.astype(np.float32)])[None]
        prob, self.state = self.sess.run(None, {"input": x, "state": self.state, "sr": self.sr})
        self.context = x[0, -self.CONTEXT:]
        return float(prob[0, 0])


class Denoiser:
    FRAME = 480  # 10 ms at 48 kHz

    def __init__(self, path=RNNOISE_PATH):
        self.lib = ctypes.CDLL(os.path.abspath(path))
        self.lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        self.lib.rnnoise_create.restype = ctypes.c_void_p
        self.lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        self.lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                                   ctypes.POINTER(ctypes.c_float)]
        self.lib.rnnoise_process_frame.restype = ctypes.c_float
        self.state = self.lib.rnnoise_create(None)
        self.pending = np.zeros(0, np.float32)

    def __call__(self, x):
        """Any number of 48 kHz samples in, the denoised samples of every complete 10 ms frame out."""
        self.pending = np.concatenate([self.pending, x.astype(np.float32)])
        n = len(self.pending) // self.FRAME * self.FRAME
        if n == 0:
            return np.zeros(0, np.float32)
        out = np.empty(n, np.float32)
        for i in range(0, n, self.FRAME):
            frame = np.ascontiguousarray(self.pending[i:i + self.FRAME] * 32768.0)  # RNNoise works in int16 scale
            ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            self.lib.rnnoise_process_frame(self.state, ptr, ptr)
            out[i:i + self.FRAME] = frame / 32768.0
        self.pending = self.pending[n:]
        return out

    def __del__(self):
        try:
            self.lib.rnnoise_destroy(self.state)
        except Exception:
            pass


def lowpass_taps(numtaps, cutoff, fs):
    """Windowed-sinc low-pass FIR (Hamming window, unit gain at 0 Hz) - scipy.signal.firwin's default design,
    without importing scipy.signal (0.7 s of the studio's start)."""
    c = cutoff / (fs / 2)
    m = np.arange(numtaps) - (numtaps - 1) / 2
    h = c * np.sinc(c * m) * np.hamming(numtaps)
    return h / h.sum()


class Resampler:
    """Streaming low-pass + resample between rates (keeps its state between chunks)."""

    def __init__(self, rate_in, rate_out=RATE):
        self.rate_in, self.step = rate_in, rate_in / rate_out
        self.b = lowpass_taps(63, 0.44 * min(rate_in, rate_out), rate_in) if rate_in != rate_out else None
        self.zi = np.zeros(62) if self.b is not None else None  # the previous chunk's last 62 input samples
        self.pos = 0.0  # position of the next output sample, in input samples, relative to the chunk start
        self.last = 0.0

    def __call__(self, x):
        if self.b is None or not len(x):
            return x.astype(np.float32)
        xx = np.concatenate([self.zi, x.astype(np.float64)])
        y = np.convolve(xx, self.b, mode="valid")  # the FIR filter, continuing from the last chunk
        self.zi = xx[-62:]
        ext = np.concatenate([[self.last], y])  # index 0 = last sample of the previous chunk
        idx = np.arange(self.pos, len(y) - 1 + 1e-9, self.step)  # only between samples we already have
        out = np.interp(idx + 1, np.arange(len(ext)), ext)
        self.pos = (idx[-1] + self.step - len(y)) if len(idx) else self.pos - len(y)
        self.last = y[-1] if len(y) else self.last
        return out.astype(np.float32)


class VoiceProcessor:
    """feed() raw microphone chunks; read clean 16 kHz audio with window() and speech with speech_prob."""

    def __init__(self, rate_in, denoise=False, vad=True):
        self.lock = threading.Lock()
        self.denoiser = None
        if denoise:
            try:
                self.denoiser = Denoiser()
            except OSError as e:  # (no RNNoise library for this OS): lip sync goes on with the voice as it is
                print(f"noise suppression unavailable ({e}); using the voice as it is", flush=True)
                denoise = False
        # RNNoise works at 48 kHz: microphones running at another rate (often 44.1 kHz) are converted first
        self.to_48k = Resampler(rate_in, 48000) if denoise and rate_in != 48000 else None
        self.resample = Resampler(48000 if denoise else rate_in)
        self.vad = SpeechDetector() if vad else None
        self.buf = np.zeros(RATE, np.float32)  # last second of clean 16 kHz audio
        self._vad_pending = np.zeros(0, np.float32)
        self.speech_prob, self.last_speech = 0.0, -1e9

    def feed(self, x):
        y = x
        if self.denoiser:
            y = self.denoiser(self.to_48k(y) if self.to_48k else y)
        y = self.resample(y)
        if not len(y):
            return
        with self.lock:
            self.buf = np.roll(self.buf, -len(y))
            self.buf[-len(y):] = y[-RATE:]
        if self.vad is not None:
            self._vad_pending = np.concatenate([self._vad_pending, y])
            while len(self._vad_pending) >= SpeechDetector.CHUNK:
                chunk, self._vad_pending = (self._vad_pending[:SpeechDetector.CHUNK],
                                            self._vad_pending[SpeechDetector.CHUNK:])
                self.speech_prob = self.vad(chunk)
                if self.speech_prob > 0.5:
                    self.last_speech = time.perf_counter()

    def window(self, n):
        with self.lock:
            return self.buf[-n:].copy()
