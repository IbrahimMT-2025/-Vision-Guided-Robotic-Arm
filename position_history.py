# position_history.py
# PositionHistory class

import math
from collections import deque
from config import POSITION_STABILITY_THRESHOLD

class PositionHistory:
    """Rolling buffer of 3-D hand positions used to detect stability."""

    def __init__(self, max_size=20):
        self.history = deque(maxlen=max_size)

    def add(self, x, y, z, timestamp):
        self.history.append({'x': x, 'y': y, 'z': z, 'time': timestamp})

    def is_stable(self, threshold=POSITION_STABILITY_THRESHOLD):
        """True if the last 5 samples span < threshold metres."""
        if len(self.history) < 5:
            return False
        recent = list(self.history)[-5:]
        x0, y0, z0 = recent[0]['x'], recent[0]['y'], recent[0]['z']
        return all(
            math.sqrt((p['x']-x0)**2 + (p['y']-y0)**2 + (p['z']-z0)**2) < threshold
            for p in recent[1:]
        )

    def get_average(self):
        """Return (x, y, z) average of last 5 samples."""
        if not self.history:
            return None, None, None
        recent = list(self.history)[-5:]
        return (
            sum(p['x'] for p in recent) / len(recent),
            sum(p['y'] for p in recent) / len(recent),
            sum(p['z'] for p in recent) / len(recent),
        )

    def clear(self):
        self.history.clear()

print('✓ PositionHistory class defined')