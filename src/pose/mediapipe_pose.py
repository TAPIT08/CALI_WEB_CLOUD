from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

from src.utils.structures import PoseResult


class MediaPipePoseEstimator:
    def __init__(
        self,
        model_complexity: int,
        smooth_landmarks: bool,
        enable_segmentation: bool,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        self.pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def estimate(self, frame: np.ndarray) -> Optional[PoseResult]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return None
        h, w = frame.shape[:2]
        landmarks = np.array(
            [
                [lm.x * w, lm.y * h, lm.z * w, lm.visibility]
                for lm in result.pose_landmarks.landmark
            ],
            dtype=np.float32,
        )
        return PoseResult(landmarks=landmarks, image_size=(w, h))

    def close(self) -> None:
        self.pose.close()
