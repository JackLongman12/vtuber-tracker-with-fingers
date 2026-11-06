"""Clap test for real motion-to-avatar delay.

The microphone hears a clap almost instantly (~10 ms); the camera shows the hands meeting some time later.
latency = (time the frame with the hands together arrived) - (time the clap was heard), measured per clap,
plus the studio's own processing time from frame arrival to sending the avatar data."""
import numpy as np


def hand_gap(res):
    """Distance between the two wrists in units of hand size (small = hands together), or None."""
    hands = res["hands"]
    if len(hands) == 2:
        a, b = hands[0]["_pts"], hands[1]["_pts"]
        size = max(np.linalg.norm(a[0, :2] - a[9, :2]), np.linalg.norm(b[0, :2] - b[9, :2]), 1.0)
        return float(np.linalg.norm(a[9, :2] - b[9, :2]) / size)  # palm centres
    body = res["body"]  # palms together often merge into one hand detection: fall back to the wrists
    if len(hands) == 1 and body is not None and body["_vis"][15] > 0.8 and body["_vis"][16] > 0.8:
        p = body["_pts"]
        shoulder = max(np.linalg.norm(p[11, :2] - p[12, :2]), 1.0)
        return float(np.linalg.norm(p[15, :2] - p[16, :2]) / shoulder * 2.5)  # roughly the same scale
    return None


class ClapLatency:
    def __init__(self, needed=6):
        self.needed, self.gaps, self.results = needed, [], []
        self.used_onsets = set()
        self.ignored = 0  # clap sounds with no clap seen on camera

    def update(self, frame_time, res, onsets):
        g = hand_gap(res)
        if g is not None:
            self.gaps.append((frame_time, g))
            self.gaps = [x for x in self.gaps if frame_time - x[0] < 3.0]
        for onset in onsets:
            if onset in self.used_onsets or frame_time - onset < 0.6:  # wait until the visual side is complete
                continue
            self.used_onsets.add(onset)
            lat = self.match(onset)
            if lat is not None:
                self.results.append(lat)
            else:
                self.ignored += 1

    def match(self, onset):
        """Frame where the hands were closest, from 0.1 s before to 0.5 s after the clap was heard. Counts only
        if the hands were clearly apart just before and came together (a real clap), not resting or jittering."""
        window = [(t, g) for t, g in self.gaps if onset - 0.1 <= t <= onset + 0.5]
        before = [(t, g) for t, g in self.gaps if onset - 0.8 <= t < onset - 0.1]
        if len(window) < 3 or len(before) < 3:
            return None
        t_min, g_min = min(window, key=lambda x: x[1])
        t_far, g_far = max(before, key=lambda x: x[1])
        if g_min > 1.0 or g_far < max(2.0, g_min + 1.5):  # hands never came together, or weren't apart first
            return None
        # closing in steadily from the widest point to contact (tracking jitter jumps back and forth)
        path = [g for t, g in self.gaps if t_far <= t <= t_min]
        if sum(b > a + 0.5 for a, b in zip(path, path[1:])) > 1:
            return None
        return t_min - onset

    def summary(self):
        if not self.results:
            return None
        r = np.array(self.results) * 1000
        return {"claps": len(r), "camera_ms": float(np.median(r)), "spread_ms": float(np.percentile(r, 75) -
                                                                                     np.percentile(r, 25))}

    @property
    def done(self):
        return len(self.results) >= self.needed
