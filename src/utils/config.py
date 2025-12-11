from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class RuntimeConfig:
    frame: Dict[str, Any]
    latency: Dict[str, Any]
    mediapipe: Dict[str, Any]
    yolo: Dict[str, Any]
    feedback: Dict[str, Any]
    display: Dict[str, Any]
    audio: Dict[str, Any]


def load_runtime_config(path: str) -> RuntimeConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return RuntimeConfig(
        frame=data.get("frame", {}),
        latency=data.get("latency", {}),
        mediapipe=data.get("mediapipe", {}),
        yolo=data.get("yolo", {}),
        feedback=data.get("feedback", {}),
        display=data.get("display", {}),
        audio=data.get("audio", {}),
    )
