from __future__ import annotations

from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from src.utils.structures import Detection


class YOLOExerciseDetector:
    def __init__(
        self,
        weights_path: str,
        conf_threshold: float,
        iou_threshold: float,
        max_det: int,
        device: str,
    ) -> None:
        self.model = YOLO(weights_path)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.device = device
        self.class_names = self.model.names

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections
        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.cls is None:
            return detections
        for cls, conf, xyxy in zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.xyxy.cpu().numpy()):
            raw_label = self.class_names.get(int(cls), str(int(cls)))
            label = raw_label.lower().replace(" ", "").replace("-", "")
            detections.append(
                Detection(
                    label=label,
                    confidence=float(conf),
                    bbox_xyxy=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                )
            )
        return detections

    def get_top_detection(self, frame: np.ndarray, interest_labels: Optional[List[str]] = None) -> Optional[Detection]:
        detections = self.detect(frame)
        if not detections:
            return None
        if interest_labels:
            detections = [d for d in detections if d.label in interest_labels]
        if not detections:
            return None
        return max(detections, key=lambda d: d.confidence)
