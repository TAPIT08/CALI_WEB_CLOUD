from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.logic.geometry import POSE_LANDMARKS
from src.utils.structures import Detection, ExerciseState, FeedbackMessage, PoseResult


POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

POSE_CONNECTION_INDEXES = [(POSE_LANDMARKS[a], POSE_LANDMARKS[b]) for a, b in POSE_CONNECTIONS]
PRIMARY_POINTS = {
    "pushup": [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    "pullup": [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    "squat": [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
}

PRIMARY_INDEXES = {
    label: {POSE_LANDMARKS[name] for name in names if name in POSE_LANDMARKS}
    for label, names in PRIMARY_POINTS.items()
}

BOX_COLORS: Dict[str, Tuple[int, int, int]] = {
    "pushup": (0, 165, 255),  # orange
    "pullup": (255, 0, 0),    # blue
    "squat": (0, 255, 0),     # green
}

JOINT_FOCUS_TIPS: Dict[str, str] = {
    "pushup": "Monitor shoulders, hips, knees, and ankles to keep push-up alignment locked in.",
    "pullup": "Track shoulders, elbows, hips, and ankles so you can spot swinging or knee tuck cheats.",
    "squat": "Watching shoulders, hips, knees, and ankles reveals squat depth and knee travel.",
}

__all__ = [
    "FrameOverlay",
    "PRIMARY_POINTS",
    "PRIMARY_INDEXES",
    "JOINT_FOCUS_TIPS",
]


class FrameOverlay:
    def __init__(
        self,
        alpha: float,
        font_scale: float,
        metric_font_scale: float,
        margin: int,
        primary_thickness: int,
        secondary_thickness: int,
    ) -> None:
        self.alpha = alpha
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.metric_font_scale = metric_font_scale
        self.margin = margin
        self.primary_thickness = max(1, primary_thickness)
        self.secondary_thickness = max(1, secondary_thickness)
        self.primary_labels = set(PRIMARY_INDEXES.keys())

    def draw(
        self,
        frame: np.ndarray,
        detection: Detection | None,
        state: ExerciseState,
        feedback: List[FeedbackMessage],
        pose: Optional[PoseResult] = None,
        show_metrics: bool = True,
        show_skeleton: bool = True,
        switch_prompt: Optional[str] = None,
        highlight_label: Optional[str] = None,
    ) -> np.ndarray:
        overlay = frame.copy()
        if show_skeleton and pose is not None:
            self._draw_skeleton(overlay, pose, highlight_label)
        if detection:
            self._draw_detection(overlay, detection)
        self._draw_state(overlay, state, show_metrics)
        self._draw_feedback(overlay, feedback, switch_prompt)
        return cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0)

    def _draw_detection(self, frame: np.ndarray, detection: Detection) -> None:
        x1, y1, x2, y2 = detection.bbox_xyxy
        color = BOX_COLORS.get(detection.label, (64, 255, 128))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.label} {detection.confidence:.2f}"
        bg = frame.copy()
        text_size, baseline = cv2.getTextSize(label, self.font, self.font_scale, 2)
        bg_top = max(0, y1 - text_size[1] - baseline - 6)
        bg_left = max(0, x1)
        cv2.rectangle(bg, (bg_left, bg_top), (bg_left + text_size[0] + 12, bg_top + text_size[1] + baseline + 6), (0, 0, 0), -1)
        cv2.addWeighted(bg, 0.45, frame, 0.55, 0, frame)
        cv2.putText(
            frame,
            label,
            (bg_left + 6, bg_top + text_size[1]),
            self.font,
            self.font_scale,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_state(self, frame: np.ndarray, state: ExerciseState, show_metrics: bool) -> None:
        hud_lines = [
            f"Exercise: {state.name}",
            f"Level: {state.level}",
            f"Phase: {state.phase}",
            f"Reps: {state.rep_count}",
        ]
        self._draw_text_block(frame, hud_lines, self.font_scale, (255, 255, 255), 2, self.margin, self.margin)
        if not show_metrics:
            return
        metrics = [f"{key}: {value:.1f}" for key, value in state.metrics.items()]
        if metrics:
            y_offset = self.margin + int(self._line_height(self.font_scale) * len(hud_lines)) + 12
            self._draw_text_block(frame, metrics, self.metric_font_scale, (180, 220, 255), 1, self.margin, y_offset)

    def _draw_feedback(
        self,
        frame: np.ndarray,
        feedback: List[FeedbackMessage],
        switch_prompt: Optional[str],
    ) -> None:
        if not feedback and not switch_prompt:
            return
        line_height = self._line_height(self.font_scale)
        lines: List[Tuple[str, Tuple[int, int, int], float]] = []
        if switch_prompt:
            lines.append((switch_prompt, (90, 255, 255), self.font_scale))
        colors: Dict[str, Tuple[int, int, int]] = {
            "info": (200, 200, 200),
            "warning": (0, 215, 255),
            "critical": (0, 0, 255),
        }
        for fb in feedback:
            color = colors.get(fb.severity, (200, 200, 200))
            lines.append((fb.message, color, self.font_scale))
            if fb.hints:
                for hint in fb.hints:
                    lines.append((f"- {hint}", color, self.metric_font_scale))

        block_height = int(sum(self._line_height(scale) for _, _, scale in lines) + 12)
        block_top = frame.shape[0] - block_height - self.margin
        block_left = self.margin
        block_right = frame.shape[1] - self.margin
        bg = frame.copy()
        cv2.rectangle(bg, (block_left, block_top), (block_right, block_top + block_height), (10, 16, 30), -1)
        cv2.addWeighted(bg, 0.45, frame, 0.55, 0, frame)

        cursor_y = block_top + 10
        for text, color, scale in lines:
            cv2.putText(
                frame,
                text,
                (block_left + 12, cursor_y + int(self._line_height(scale) * 0.75)),
                self.font,
                scale,
                color,
                2 if scale >= self.font_scale else 1,
                cv2.LINE_AA,
            )
            cursor_y += self._line_height(scale)

    def _draw_skeleton(
        self,
        frame: np.ndarray,
        pose: PoseResult,
        highlight_label: Optional[str],
    ) -> None:
        landmarks = pose.landmarks
        primary_indexes = PRIMARY_INDEXES.get(highlight_label or "", set())
        primary_color = (72, 199, 239)
        secondary_color = (120, 120, 120)
        for start_idx, end_idx in POSE_CONNECTION_INDEXES:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            if start[3] < 0.5 or end[3] < 0.5:
                continue
            both_primary = start_idx in primary_indexes and end_idx in primary_indexes
            color = primary_color if both_primary else secondary_color
            thickness = self.primary_thickness if both_primary else self.secondary_thickness
            cv2.line(
                frame,
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                color,
                thickness,
            )
        for idx, point in enumerate(landmarks):
            if point[3] < 0.5:
                continue
            is_primary = idx in primary_indexes
            color = (255, 255, 255) if is_primary else (130, 130, 130)
            radius = 5 if is_primary else 3
            cv2.circle(frame, (int(point[0]), int(point[1])), radius, color, -1)

    def _draw_text_block(
        self,
        frame: np.ndarray,
        lines: List[str],
        scale: float,
        color: Tuple[int, int, int],
        thickness: int,
        x: int,
        y: int,
    ) -> None:
        if not lines:
            return
        line_height = self._line_height(scale)
        max_width = 0
        for line in lines:
            width, _ = cv2.getTextSize(line, self.font, scale, thickness)[0]
            max_width = max(max_width, width)
        top = y
        left = x
        bottom = top + line_height * len(lines) + 12
        right = left + max_width + 24
        bg = frame.copy()
        cv2.rectangle(bg, (left - 12, top - 8), (right, bottom), (10, 16, 30), -1)
        cv2.addWeighted(bg, 0.45, frame, 0.55, 0, frame)
        for idx, line in enumerate(lines):
            baseline = top + 12 + idx * line_height
            cv2.putText(
                frame,
                line,
                (left, baseline),
                self.font,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

    def _line_height(self, scale: float) -> int:
        return max(18, int(26 * scale))
