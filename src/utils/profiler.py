from __future__ import annotations

import time
from collections import deque
from typing import Deque


class FPSMeter:
    def __init__(self, window: int = 30) -> None:
        self.window = max(1, window)
        self.buffer: Deque[float] = deque(maxlen=self.window)
        self.last_time = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        delta = now - self.last_time
        self.last_time = now
        if delta == 0:
            fps = 0.0
        else:
            fps = 1.0 / delta
        self.buffer.append(fps)
        return fps

    def get_fps(self) -> float:
        if not self.buffer:
            return 0.0
        return float(sum(self.buffer) / len(self.buffer))
