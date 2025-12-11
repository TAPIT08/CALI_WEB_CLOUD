from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import yaml

from src.logic.geometry import angle_3pts, get_landmarks_map, vertical_ratio
from src.utils.structures import ExerciseState, FeedbackMessage, PoseResult


class RuleBasedCoach:
    def __init__(self, levels_config_path: str, level: str, smoothing_window: int) -> None:
        with open(levels_config_path, "r", encoding="utf-8") as f:
            self.thresholds = yaml.safe_load(f)
        if level not in {"beginner", "intermediate", "advanced"}:
            raise ValueError(f"Unsupported level: {level}")
        self.level = level
        self.smoothing_window = max(1, smoothing_window)
        self.states: Dict[str, Dict[str, object]] = {}

    def update(self, exercise: str, pose: PoseResult) -> Tuple[ExerciseState, List[FeedbackMessage]]:
        if exercise not in self.thresholds:
            return self._default_state(exercise), []
        if exercise not in self.states:
            self.states[exercise] = self._init_state(exercise)
        state_data = self.states[exercise]
        landmarks = get_landmarks_map(pose.landmarks)
        metrics, feedback = self._compute_metrics(exercise, landmarks, pose.image_size)
        metrics_history: Deque[Dict[str, float]] = state_data["history"]
        metrics_history.append(metrics)
        smoothed = self._smooth_metrics(metrics_history)
        phase, rep_count, extra_feedback = self._update_phase(exercise, state_data, smoothed)
        feedback.extend(extra_feedback)
        return (
            ExerciseState(
                name=exercise,
                phase=phase,
                rep_count=rep_count,
                metrics=smoothed,
                level=self.level,
            ),
            feedback,
        )

    def _init_state(self, exercise: str) -> Dict[str, object]:
        return {
            "phase": "top",
            "rep_count": 0,
            "history": deque(maxlen=self.smoothing_window),
            "frame_in_phase": 0,
        }

    def _default_state(self, exercise: str) -> ExerciseState:
        return ExerciseState(name=exercise, phase="idle", rep_count=0, metrics={}, level=self.level)

    def _smooth_metrics(self, history: Deque[Dict[str, float]]) -> Dict[str, float]:
        if not history:
            return {}
        keys = history[0].keys()
        return {k: float(np.mean([entry[k] for entry in history])) for k in keys}

    def _update_phase(
        self,
        exercise: str,
        state_data: Dict[str, object],
        metrics: Dict[str, float],
    ) -> Tuple[str, int, List[FeedbackMessage]]:
        phase: str = state_data["phase"]  # type: ignore[assignment]
        rep_count: int = state_data["rep_count"]  # type: ignore[assignment]
        frame_in_phase: int = state_data["frame_in_phase"]  # type: ignore[assignment]
        exercise_config = self.thresholds[exercise][self.level]
        feedback: List[FeedbackMessage] = []
        tempo_min = int(exercise_config.get("tempo_min_frames", 0))

        if exercise == "pushup":
            down_thresh = exercise_config["elbow_down_angle"]
            up_thresh = exercise_config["elbow_up_angle"]
            if metrics["elbow_angle"] <= down_thresh and phase != "bottom":
                if frame_in_phase < tempo_min and frame_in_phase > 0:
                    feedback.append(FeedbackMessage("Control the descent; slow it down.", "info"))
                phase = "bottom"
                frame_in_phase = 0
            elif metrics["elbow_angle"] >= up_thresh and phase == "bottom":
                if frame_in_phase < tempo_min and frame_in_phase > 0:
                    feedback.append(FeedbackMessage("Pause briefly at the bottom for better control.", "info"))
                phase = "top"
                rep_count += 1
                frame_in_phase = 0
            else:
                frame_in_phase += 1
        elif exercise == "pullup":
            bottom_thresh = exercise_config["elbow_bottom_angle"]
            top_thresh = exercise_config["elbow_top_angle"]
            if metrics["elbow_angle"] >= bottom_thresh and phase != "bottom":
                if frame_in_phase < tempo_min and frame_in_phase > 0:
                    feedback.append(FeedbackMessage("Lower under control; avoid dropping.", "info"))
                phase = "bottom"
                frame_in_phase = 0
            elif metrics["elbow_angle"] <= top_thresh and phase == "bottom":
                if frame_in_phase < tempo_min and frame_in_phase > 0:
                    feedback.append(FeedbackMessage("Squeeze at the top for a moment.", "info"))
                phase = "top"
                rep_count += 1
                frame_in_phase = 0
            else:
                frame_in_phase += 1
        elif exercise == "squat":
            bottom_thresh = exercise_config["knee_bottom_angle"]
            top_thresh = exercise_config["knee_top_angle"]
            if metrics["knee_angle"] <= bottom_thresh and phase != "bottom":
                if frame_in_phase < tempo_min and frame_in_phase > 0:
                    feedback.append(FeedbackMessage("Sit back gradually; control the descent.", "info"))
                phase = "bottom"
                frame_in_phase = 0
            elif metrics["knee_angle"] >= top_thresh and phase == "bottom":
                if frame_in_phase < tempo_min and frame_in_phase > 0:
                    feedback.append(FeedbackMessage("Drive up with control; avoid bouncing.", "info"))
                phase = "top"
                rep_count += 1
                frame_in_phase = 0
            else:
                frame_in_phase += 1

        state_data["phase"] = phase
        state_data["rep_count"] = rep_count
        state_data["frame_in_phase"] = frame_in_phase

        return phase, rep_count, feedback

    def _compute_metrics(
        self,
        exercise: str,
        lm: Dict[str, np.ndarray],
        image_size: Tuple[int, int],
    ) -> Tuple[Dict[str, float], List[FeedbackMessage]]:
        width, height = image_size
        feedback: List[FeedbackMessage] = []
        metrics: Dict[str, float] = {}
        config = self.thresholds[exercise][self.level]

        if exercise == "pushup":
            elbow_angle = self._avg_elbow_angle(lm)
            hip_angle = self._avg_angle(lm, "shoulder", "hip", "knee")
            hip_drop = self._hip_drop_ratio(lm, height)
            metrics = {
                "elbow_angle": elbow_angle,
                "hip_angle": hip_angle,
                "hip_drop": hip_drop,
            }
            if hip_angle < config["hip_angle_min"]:
                feedback.append(FeedbackMessage("Keep hips aligned with shoulders.", "warning"))
            if hip_drop > config["hip_drop_max_ratio"]:
                feedback.append(FeedbackMessage("Avoid hip sag; tighten your core.", "warning"))
            if elbow_angle > config["elbow_down_angle"] * 1.1:
                feedback.append(FeedbackMessage("Go lower to reach full depth.", "info"))
        elif exercise == "pullup":
            elbow_angle = self._avg_elbow_angle(lm)
            chin_ratio = self._chin_over_bar_ratio(lm, height)
            shoulder_ratio = self._shoulder_shrug_ratio(lm, height)
            metrics = {
                "elbow_angle": elbow_angle,
                "chin_ratio": chin_ratio,
                "shoulder_ratio": shoulder_ratio,
            }
            if chin_ratio > config["chin_over_bar_ratio"]:
                feedback.append(FeedbackMessage("Pull higher until chin clears the bar.", "warning"))
            if shoulder_ratio > config["shoulder_engagement_ratio"]:
                feedback.append(FeedbackMessage("Drive shoulders down; avoid shrugging.", "info"))
        elif exercise == "squat":
            knee_angle = self._avg_knee_angle(lm)
            hip_depth = self._hip_depth_ratio(lm, height)
            knee_valgus = self._knee_valgus_ratio(lm, width)
            torso_forward = self._torso_forward_angle(lm)
            metrics = {
                "knee_angle": knee_angle,
                "hip_depth": hip_depth,
                "knee_valgus": knee_valgus,
                "torso_forward": torso_forward,
            }
            if knee_angle > config["knee_bottom_angle"] * 1.05:
                feedback.append(FeedbackMessage("Sit deeper to hit parallel.", "warning"))
            if hip_depth > config["hip_depth_ratio"]:
                feedback.append(FeedbackMessage("Drive hips back; keep weight on heels.", "info"))
            if knee_valgus > config["knee_valgus_ratio"]:
                feedback.append(FeedbackMessage("Push knees out to avoid valgus.", "warning"))
            if torso_forward > config["torso_forward_angle"]:
                feedback.append(FeedbackMessage("Keep chest up; hinge less.", "info"))
        else:
            metrics = {}

        return metrics, feedback

    def _avg_elbow_angle(self, lm: Dict[str, np.ndarray]) -> float:
        left = angle_3pts(lm["left_shoulder"], lm["left_elbow"], lm["left_wrist"])
        right = angle_3pts(lm["right_shoulder"], lm["right_elbow"], lm["right_wrist"])
        return float((left + right) / 2)

    def _avg_angle(self, lm: Dict[str, np.ndarray], a: str, b: str, c: str) -> float:
        left = angle_3pts(lm[f"left_{a}"], lm[f"left_{b}"], lm[f"left_{c}"])
        right = angle_3pts(lm[f"right_{a}"], lm[f"right_{b}"], lm[f"right_{c}"])
        return float((left + right) / 2)

    def _hip_drop_ratio(self, lm: Dict[str, np.ndarray], height: int) -> float:
        shoulder_y = (lm["left_shoulder"][1] + lm["right_shoulder"][1]) / 2
        hip_y = (lm["left_hip"][1] + lm["right_hip"][1]) / 2
        return float(abs(hip_y - shoulder_y) / max(height, 1))

    def _chin_over_bar_ratio(self, lm: Dict[str, np.ndarray], height: int) -> float:
        nose = lm["nose"]
        left_wrist = lm["left_wrist"]
        right_wrist = lm["right_wrist"]
        bar_y = min(left_wrist[1], right_wrist[1])
        return vertical_ratio(nose, np.array([nose[0], bar_y, nose[2], nose[3]]), height)

    def _shoulder_shrug_ratio(self, lm: Dict[str, np.ndarray], height: int) -> float:
        left = vertical_ratio(lm["left_shoulder"], lm["left_ear"], height)
        right = vertical_ratio(lm["right_shoulder"], lm["right_ear"], height)
        return float((abs(left) + abs(right)) / 2)

    def _avg_knee_angle(self, lm: Dict[str, np.ndarray]) -> float:
        left = angle_3pts(lm["left_hip"], lm["left_knee"], lm["left_ankle"])
        right = angle_3pts(lm["right_hip"], lm["right_knee"], lm["right_ankle"])
        return float((left + right) / 2)

    def _hip_depth_ratio(self, lm: Dict[str, np.ndarray], height: int) -> float:
        hip_y = (lm["left_hip"][1] + lm["right_hip"][1]) / 2
        knee_y = (lm["left_knee"][1] + lm["right_knee"][1]) / 2
        return float((hip_y - knee_y) / max(height, 1))

    def _knee_valgus_ratio(self, lm: Dict[str, np.ndarray], width: int) -> float:
        knee_dist = abs(lm["left_knee"][0] - lm["right_knee"][0])
        ankle_dist = abs(lm["left_ankle"][0] - lm["right_ankle"][0])
        if ankle_dist == 0:
            return 0.0
        return float((ankle_dist - knee_dist) / max(width, 1))

    def _torso_forward_angle(self, lm: Dict[str, np.ndarray]) -> float:
        left_shoulder = lm["left_shoulder"]
        left_hip = lm["left_hip"]
        vec = left_shoulder[:2] - left_hip[:2]
        if np.linalg.norm(vec) == 0:
            return 0.0
        vertical = np.array([0.0, -1.0])
        cosine = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical))
        cosine = np.clip(cosine, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))
