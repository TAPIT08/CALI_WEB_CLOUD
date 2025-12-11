from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


POSE_LANDMARKS = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


def angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0.0
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def vertical_ratio(a: np.ndarray, b: np.ndarray, image_height: int) -> float:
    return float((a[1] - b[1]) / max(image_height, 1))


def horizontal_distance(a: np.ndarray, b: np.ndarray, image_width: int) -> float:
    return float(abs(a[0] - b[0]) / max(image_width, 1))


def average_visibility(points: Iterable[np.ndarray]) -> float:
    visibilities = [p[3] for p in points if p is not None]
    if not visibilities:
        return 0.0
    return float(sum(visibilities) / len(visibilities))


def get_landmarks_map(landmarks: np.ndarray) -> Dict[str, np.ndarray]:
    return {name: landmarks[idx] for name, idx in POSE_LANDMARKS.items()}
