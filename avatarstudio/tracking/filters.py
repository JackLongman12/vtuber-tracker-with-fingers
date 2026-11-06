"""One Euro filter (Casiez et al. 2012): smooths jitter when still, follows quickly when moving."""
import math

import numpy as np


class OneEuro:
    """Filters a fixed-size float vector. min_cutoff: lower = smoother when still (Hz);
    beta: higher = less lag when moving fast."""

    def __init__(self, min_cutoff=1.0, beta=0.5, d_cutoff=1.0, lead=0.0):
        """lead: predict this many seconds ahead along the current motion (hides camera + processing delay)."""
        self.min_cutoff, self.beta, self.d_cutoff, self.lead = min_cutoff, beta, d_cutoff, lead
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self.x is None or self.x.shape != x.shape:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x + self.dx * self.lead if self.lead else self.x.copy()


class FilterBank:
    """One OneEuro per named signal, created on first use."""

    def __init__(self, **defaults):
        self.defaults, self.filters = defaults, {}

    def __call__(self, name, x, t, **params):
        if name not in self.filters:
            self.filters[name] = OneEuro(**{**self.defaults, **params})
        f = self.filters[name]
        for k, v in params.items():  # settings can change while running
            setattr(f, k, v)
        return f(x, t)
