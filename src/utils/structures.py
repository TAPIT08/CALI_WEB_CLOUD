from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]


@dataclass
class PoseResult:
    landmarks: np.ndarray  # shape: (33, 4) -> x, y, z, visibility
    image_size: Tuple[int, int]

    def get_landmark(self, idx: int) -> Optional[np.ndarray]:
        if 0 <= idx < self.landmarks.shape[0]:
            return self.landmarks[idx]
        return None


@dataclass
class FeedbackMessage:
    message: str
    severity: str  # info | warning | critical
    hints: Optional[List[str]] = None


@dataclass
class ExerciseState:
    name: str
    phase: str
    rep_count: int
    metrics: Dict[str, float]
    level: str