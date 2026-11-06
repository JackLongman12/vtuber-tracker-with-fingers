"""Microphone lip sync: loudness says how far the mouth opens, the voice's formants say which vowel.

Formants are the resonances of your mouth and throat; the first two (F1, F2) separate the five vowels:
A = high F1, I = low F1 + high F2, U = low F1 + mid F2, E = mid F1 + high F2, O = mid F1 + low F2.
They're estimated with linear prediction (LPC) on 30 ms of audio. The V calibration also records your own
voice, so matching uses your formants instead of the textbook averages below.
"""
import math
import queue
import threading
import time

import numpy as np

from avatarstudio.audio.voice import VoiceProcessor

RATE = 16000
WINDOW = 480  # 30 ms
VOWELS = ("A", "I", "U", "E", "O")
# typical adult formants (Hz) for A I U E O; used until you run the V calibration
DEFAULT_FORMANTS = {"A": (780, 1300), "I": (310, 2250), "U": (340, 1250), "E": (500, 1850), "O": (500, 870)}


def lpc(x, order):
    """Linear prediction coefficients (Levinson-Durbin on the autocorrelation)."""
    r = np.correlate(x, x, "full")[len(x) - 1:len(x) + order]
    if r[0] <= 0:
        return None
    a, err = np.zeros(order + 1), r[0]
    a[0] = 1.0
    for i in range(1, order + 1):
        k = -(r[i] + np.dot(a[1:i], r[i - 1:0:-1])) / err
        a[1:i + 1] = a[1:i + 1] + k * np.append(a[i - 1:0:-1], 1.0)
        err *= 1 - k * k
        if err <= 0:
            return None
    return a


def formants(x, rate=RATE):
    """First two formant frequencies (Hz) of a voiced frame, or None."""
    x = np.append(x[0], x[1:] - 0.97 * x[:-1]) * np.hamming(len(x))  # pre-emphasis + window
    a = lpc(x, 2 + rate // 1000)
    if a is None:
        return None
    roots = [r for r in np.roots(a) if r.imag > 0]
    found = []
    for r in roots:
        f = math.atan2(r.imag, r.real) * rate / (2 * math.pi)
        bw = -0.5 * rate / math.pi * math.log(abs(r))
        if 200 < f < 3500 and bw < 500:
            found.append(f)
    found.sort()
    return tuple(found[:2]) if len(found) >= 2 else None


class MicLipSync:
    """Reads the microphone in the background; read() gives the current loudness and vowel weights."""

    def __init__(self, device=None, voice=None, vad=True, denoise=False):
        """vad: only open the mouth for human speech (voice.SpeechDetector). denoise: noise suppression
        (voice.Denoiser) on the voice before lip sync; claps for the latency test are detected on raw audio."""
        self.device, self.use_vad, self.use_denoise = device, vad, denoise
        self.queue = queue.Queue(maxsize=50)  # 0.5 s of audio; if processing falls behind, the oldest is dropped
        self.gate, self.running = 0.0, True
        self.onsets = []  # perf_counter times of claps, for the latency test
        self.sounds_rejected = 0  # sudden sounds that weren't clap-like (speech, typing, bumps)
        self._block_db, self._floor_db = -100.0, -60.0
        self._cand = None  # clap candidate being checked: [time, peak dB, seconds since]
        self._quiet_until = 0.0
        self._open()
        self.floor_db, self.peak_db, self.t = -60.0, -20.0, time.time()
        self.level, self.history, self.last_retry, self.restarts = 0.0, [], 0.0, 0
        self.set_voice(voice)
        threading.Thread(target=self._process_loop, daemon=True).start()

    def _open(self):
        import sounddevice as sd
        # noise suppression works at 48 kHz; otherwise 16 kHz is all lip sync needs
        rates = ([48000] if self.use_denoise else []) + [RATE]
        self.stream = None
        for rate in rates:
            try:
                self.stream = sd.InputStream(samplerate=rate, channels=1, device=self.device, callback=self._cb,
                                             blocksize=rate // 100)
                self.rate = rate
                break
            except Exception:
                continue
        if self.stream is None:  # device can't do either itself: record at its own rate and resample
            self.rate = int(sd.query_devices(self.device, "input")["default_samplerate"])
            self.stream = sd.InputStream(samplerate=self.rate, channels=1, device=self.device, callback=self._cb,
                                         blocksize=self.rate // 100)
        try:
            self.proc = VoiceProcessor(self.rate, denoise=self.use_denoise, vad=self.use_vad)
        except Exception as e:  # model files missing: plain lip sync
            print(f"speech detection / noise suppression unavailable ({e})")
            self.proc = VoiceProcessor(self.rate, denoise=False, vad=False)
        self.stream.start()

    def _process_loop(self):
        """Noise suppression and speech detection run here, off the audio thread, in arrival order."""
        while self.running:
            try:
                x = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.proc.feed(x)
            except Exception as e:
                print(f"voice processing error: {e}")

    def _check_stream(self):
        """Microphone unplugged / stream died: try to reopen it every 2 s."""
        if self.stream.active or time.time() - self.last_retry < 2.0:
            return
        self.last_retry = time.time()
        try:
            self.stream.close()
        except Exception:
            pass
        try:
            import sounddevice as sd
            sd._terminate()
            sd._initialize()  # re-scan devices so a re-plugged microphone is found
            self._open()
            self.restarts += 1
        except Exception:
            pass

    def _cb(self, data, frames, t, status):
        x = data[:, 0]
        db = 20 * math.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-9)
        self._detect_clap(x, db, frames)
        self._block_db = 0.7 * self._block_db + 0.3 * db
        try:
            self.queue.put_nowait(x.copy())
        except queue.Full:  # lip sync must stay current rather than catch up on old audio
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(x.copy())

    def _detect_clap(self, x, db, frames):
        """A clap is sudden (20 dB over the background within one 10 ms block), bright (much of its energy
        above 1.5 kHz, unlike voiced speech) and short (gone again within 150 ms, unlike a word starting)."""
        now, dt = time.perf_counter(), frames / self.rate
        c = self._cand
        if c is not None:
            c[1], c[2] = max(c[1], db), c[2] + dt
            if db < c[1] - 15:  # died away quickly: a clap
                self.onsets.append(c[0])
                del self.onsets[:-50]
                self._cand = None
            elif c[2] > 0.15:  # still loud: speech, music, a long noise
                self.sounds_rejected += 1
                self._cand = None
        elif now > self._quiet_until and db > self._floor_db + 20 and db > -45:
            spec = np.abs(np.fft.rfft(x)) ** 2
            freqs = np.fft.rfftfreq(len(x), 1 / self.rate)
            bright = spec[freqs >= 1500].sum() / max(spec.sum(), 1e-12)
            if bright > 0.25:
                self._cand = [now - dt, db, 0.0]
            else:
                self.sounds_rejected += 1
            self._quiet_until = now + 0.25  # one sound, one candidate
        if db > self._floor_db + 10:  # re-arm only once the sound is over
            self._quiet_until = max(self._quiet_until, now + 0.05)
        # background level: follows quiet moments quickly, loud ones only slowly
        self._floor_db += (db - self._floor_db) * (0.2 if db < self._floor_db else 0.002)

    def set_voice(self, voice):
        """voice: {vowel: (F1, F2)} from the calibration, else textbook averages."""
        protos = voice if voice and all(v in voice for v in VOWELS) else DEFAULT_FORMANTS
        self.protos = {v: np.log(np.array(protos[v], float)) for v in VOWELS}
        gaps = [np.sum((self.protos[a] - self.protos[b]) ** 2) for a in VOWELS for b in VOWELS if a < b]
        self.sigma2 = max(min(gaps) * 0.5, 0.01)

    def _window(self):
        return self.proc.window(WINDOW).astype(np.float64)  # clean voice, already at 16 kHz

    def _speech_gate(self, dt):
        """1 while speech was heard in the last 0.3 s (words have short pauses), easing to 0 after."""
        if self.proc.vad is None:
            return 1.0
        target = 1.0 if time.perf_counter() - self.proc.last_speech < 0.3 else 0.0
        self.gate += (target - self.gate) * min(1.0, dt / (0.02 if target > self.gate else 0.1))
        return self.gate

    def formants_now(self):
        x = self._window()
        return formants(x) if np.sqrt(np.mean(x ** 2)) > 1e-4 else None

    def read(self):
        """{'level': 0..1 voice loudness above the background, 'vowels': {A..O: weight, sums to 1} or {}}"""
        self._check_stream()
        x = self._window()
        now = time.time()
        dt, self.t = now - self.t, now
        db = 20 * math.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
        # background level = the quietest 15% of the last ~6 s (speech comes in bursts, so this stays on the room
        # noise); needs ~0.5 s of history before it trusts anything
        self.history.append((now, db))
        while self.history and now - self.history[0][0] > 6.0:
            self.history.pop(0)
        if now - self.history[0][0] < 0.5:
            return {"level": 0.0, "vowels": {}, "db": db, "speech": self.proc.speech_prob}
        self.floor_db = float(np.percentile([d for _, d in self.history], 15))
        self.peak_db = max(self.peak_db - dt * 2.0, db, self.floor_db + 20)  # loudest recent speech
        level = float(np.clip((db - self.floor_db - 6) / (self.peak_db - self.floor_db - 6), 0, 1))
        level *= self._speech_gate(dt)  # not a voice (typing, fan, game audio): the mouth stays shut
        self.level += (level - self.level) * (0.6 if level > self.level else 0.3)  # fast attack, softer release
        vowels = {}
        if self.level > 0.1:
            f = formants(x)
            if f:
                p = np.log(np.array(f))
                w = np.array([math.exp(-np.sum((p - self.protos[v]) ** 2) / self.sigma2) for v in VOWELS])
                if w.sum() > 1e-6:
                    vowels = {v: float(x_) for v, x_ in zip(VOWELS, w / w.sum())}
        return {"level": self.level, "vowels": vowels, "db": db, "speech": self.proc.speech_prob}

    def close(self):
        self.running = False
        try:  # (an unplugged microphone's stream can refuse to stop; closing must not fail)
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


def fuse(camera, mic, mic_weight=0.5):
    """Blend camera vowels with voice vowels. While you're speaking, the voice decides how far the mouth opens
    (the camera can miss syllables) and shares the say in which vowel; when you're silent it's camera only."""
    if not mic:
        return camera
    speaking = mic["level"]
    cam_total = sum(camera.values())
    open_amount = min(max(cam_total, speaking), 1.0)
    if open_amount < 1e-3:
        return {v: 0.0 for v in VOWELS}
    cam_mix = {v: camera.get(v, 0.0) / cam_total for v in VOWELS} if cam_total > 1e-3 else None
    mix = speaking * mic_weight if mic["vowels"] else 0.0
    shape = {}
    for v in VOWELS:
        c = cam_mix[v] if cam_mix else mic["vowels"].get(v, 0.2)
        shape[v] = (1 - mix) * c + mix * mic["vowels"].get(v, 0.0)
    total = sum(shape.values()) or 1.0
    return {v: shape[v] / total * open_amount for v in VOWELS}
