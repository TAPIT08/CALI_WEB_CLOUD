from __future__ import annotations

from typing import List

import cv2


def enumerate_cameras(max_devices: int = 6) -> List[int]:
    indices: List[int] = []
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            indices.append(idx)
            cap.release()
        else:
            cap.release()
    return indices
