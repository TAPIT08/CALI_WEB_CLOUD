from __future__ import annotations

import asyncio
import base64
import datetime as dt
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import cv2
import numpy as np
from loguru import logger

from src.detectors.yolo_detector import YOLOExerciseDetector
from src.logic.rules import RuleBasedCoach
from src.pose.mediapipe_pose import MediaPipePoseEstimator
from src.ui.overlay import FrameOverlay, JOINT_FOCUS_TIPS, PRIMARY_INDEXES
from src.utils.camera import enumerate_cameras
from src.utils.config import load_runtime_config
from src.utils.profiler import FPSMeter
from src.utils.structures import Detection, ExerciseState, FeedbackMessage, PoseResult

from .database import record_session

FIXED_WEIGHTS_PATH = Path("weights/yolov8n-exercise.pt")
POSE_VISIBILITY_THRESHOLD = 0.5
DEFAULT_RUNTIME_CONFIG = Path("configs/runtime.yaml")
DEFAULT_LEVELS_CONFIG = Path("configs/exercise_levels.yaml")


@dataclass
class SessionConfig:
    exercise: str = "all"
    level: str = "beginner"
    camera_index: int = -1
    detection_stride: int = 3
    pose_stride: int = 1
    show_skeleton: bool = True
    show_metrics: bool = True
    goal_reps: int = 20


@dataclass
class SessionSummary:
    user_id: str
    exercise: str
    rep_count: int
    warnings: int
    started_at: dt.datetime
    ended_at: dt.datetime
    session_metadata: Dict[str, object]


class SessionAlreadyRunningError(RuntimeError):
    pass


class SessionNotRunningError(RuntimeError):
    pass


class ExerciseSession:
    """Async-aware manager that runs the CV pipeline and streams frames via an asyncio queue."""

    def __init__(
        self,
        runtime_config_path: Path | str = DEFAULT_RUNTIME_CONFIG,
        levels_config_path: Path | str = DEFAULT_LEVELS_CONFIG,
    ) -> None:
        self.runtime_config_path = Path(runtime_config_path)
        self.levels_config_path = Path(levels_config_path)
        self._async_lock = asyncio.Lock()
        self._thread_lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._frame_queue: Optional[asyncio.Queue[Dict[str, object]]] = None
        self._stop_event = threading.Event()
        self._running_user: Optional[str] = None
        self._latest_payload: Optional[Dict[str, object]] = None
        self._status: Dict[str, object] = {"running": False, "message": "idle"}
        self._summary: Optional[SessionSummary] = None

    async def start(self, user_id: str, config: SessionConfig) -> None:
        async with self._async_lock:
            if self._running_user is not None:
                raise SessionAlreadyRunningError("A session is already running")
            if not FIXED_WEIGHTS_PATH.is_file():
                raise FileNotFoundError(
                    f"Expected YOLO weights at {FIXED_WEIGHTS_PATH}. Please place the file before running."
                )
            self._stop_event.clear()
            self._frame_queue = asyncio.Queue(maxsize=2)
            self._running_user = user_id
            self._loop = asyncio.get_running_loop()
            self._summary = None
            self._latest_payload = None
            with self._thread_lock:
                self._thread = threading.Thread(
                    target=self._run_loop,
                    name="exercise-session",
                    args=(self._loop, user_id, config),
                    daemon=True,
                )
                self._thread.start()
            logger.info("Session started for user {} exercise={} level={}", user_id, config.exercise, config.level)

    async def stop(self, user_id: Optional[str] = None) -> None:
        async with self._async_lock:
            if self._running_user is None:
                raise SessionNotRunningError("No active session")
            if user_id and self._running_user != user_id:
                raise SessionNotRunningError("A different user session is active")
            self._stop_event.set()
            thread = None
            with self._thread_lock:
                thread = self._thread
            if thread:
                thread.join(timeout=5)
            logger.info("Session stop requested by user {}", user_id or self._running_user)
            self._running_user = None
            self._thread = None
            self._frame_queue = None

    def is_running(self) -> bool:
        return self._running_user is not None and not self._stop_event.is_set()

    def get_status(self) -> Dict[str, object]:
        status = dict(self._status)
        if self._latest_payload:
            status.setdefault("fps", self._latest_payload.get("fps"))
            state = self._latest_payload.get("state", {}) if isinstance(self._latest_payload, dict) else {}
            status.setdefault("repCount", state.get("rep_count"))
        if self._running_user:
            status["userId"] = self._running_user
        return status

    async def frame_generator(self, user_id: str) -> AsyncGenerator[Dict[str, object], None]:
        if self._running_user != user_id:
            raise SessionNotRunningError("No streaming session for this user")
        assert self._frame_queue is not None
        while True:
            payload = await self._frame_queue.get()
            yield payload
            if not payload.get("running", True):
                break

    def _run_loop(self, loop: asyncio.AbstractEventLoop, user_id: str, config: SessionConfig) -> None:
        runtime_cfg = None
        cap: Optional[cv2.VideoCapture] = None
        pose_estimator: Optional[MediaPipePoseEstimator] = None
        detector: Optional[YOLOExerciseDetector] = None
        summary: Optional[SessionSummary] = None
        resolved_camera_index: Optional[int] = None
        try:
            runtime_cfg = load_runtime_config(str(self.runtime_config_path))
            available_cameras = enumerate_cameras()
            camera_index = config.camera_index
            if camera_index < 0:
                camera_index = available_cameras[0] if available_cameras else 0
            resolved_camera_index = camera_index
            device_pref = runtime_cfg.latency.get("use_gpu", "auto")
            device = self._resolve_device(device_pref)
            detector = YOLOExerciseDetector(
                weights_path=str(FIXED_WEIGHTS_PATH),
                conf_threshold=runtime_cfg.yolo.get("conf_threshold", 0.35),
                iou_threshold=runtime_cfg.yolo.get("iou_threshold", 0.5),
                max_det=runtime_cfg.yolo.get("max_det", 5),
                device=device,
            )
            pose_estimator = MediaPipePoseEstimator(
                model_complexity=int(runtime_cfg.mediapipe.get("model_complexity", 1)),
                smooth_landmarks=bool(runtime_cfg.mediapipe.get("smooth_landmarks", True)),
                enable_segmentation=bool(runtime_cfg.mediapipe.get("enable_segmentation", False)),
                min_detection_confidence=float(runtime_cfg.mediapipe.get("min_detection_confidence", 0.5)),
                min_tracking_confidence=float(runtime_cfg.mediapipe.get("min_tracking_confidence", 0.5)),
            )
            coach = RuleBasedCoach(
                levels_config_path=str(self.levels_config_path),
                level=config.level,
                smoothing_window=int(runtime_cfg.latency.get("smoothing_window", 5)),
            )
            overlay = FrameOverlay(
                alpha=float(runtime_cfg.latency.get("overlay_alpha", 0.75)),
                font_scale=float(runtime_cfg.display.get("font_scale", 0.6)),
                metric_font_scale=float(runtime_cfg.display.get("metric_font_scale", 0.5)),
                margin=int(runtime_cfg.display.get("hud_margin", 16)),
                primary_thickness=int(runtime_cfg.display.get("skeleton_primary_thickness", 3)),
                secondary_thickness=int(runtime_cfg.display.get("skeleton_secondary_thickness", 1)),
            )
            fps_meter = FPSMeter()
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera index {camera_index}.")
            self._configure_capture(cap, runtime_cfg.frame)
            all_exercises = list(coach.thresholds.keys())
            interest_labels = all_exercises[:] if config.exercise == "all" else [config.exercise]
            default_state_name = config.exercise if config.exercise != "all" else "idle"
            exercise_state = ExerciseState(
                name=default_state_name,
                phase="idle",
                rep_count=0,
                metrics={},
                level=config.level,
            )
            current_detection: Optional[Detection] = None
            last_pose: Optional[PoseResult] = None
            joint_tip_announced: set[str] = set()
            frame_index = 0
            detection_stride = max(1, config.detection_stride)
            pose_stride = max(1, config.pose_stride)
            detection_cooldown = detection_stride * 2
            detection_miss_count = 0
            last_visibility_prompt = 0.0
            target_width = int(runtime_cfg.frame.get("target_width", 960))
            warnings_total = 0
            start_time = dt.datetime.utcnow()
            summary = SessionSummary(
                user_id=user_id,
                exercise=config.exercise,
                rep_count=0,
                warnings=0,
                started_at=start_time,
                ended_at=start_time,
                session_metadata={"goal_reps": config.goal_reps},
            )
            self._status = {
                "running": True,
                "message": "session running",
                "exercise": config.exercise,
                "level": config.level,
                "camera": camera_index,
                "goalReps": config.goal_reps,
            }
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                if runtime_cfg.frame.get("flip", False):
                    frame = cv2.flip(frame, 1)
                if runtime_cfg.frame.get("mirror", False):
                    frame = cv2.flip(frame, 0)
                scale = target_width / frame.shape[1]
                resized = cv2.resize(frame, (target_width, int(frame.shape[0] * scale)))
                if frame_index % detection_stride == 0:
                    detections = detector.detect(resized)
                    current_detection = self._select_detection(detections, interest_labels)
                    if current_detection is None:
                        detection_miss_count += 1
                        if detection_miss_count >= detection_cooldown:
                            exercise_state = ExerciseState(
                                name=default_state_name,
                                phase="idle",
                                rep_count=exercise_state.rep_count,
                                metrics={},
                                level=config.level,
                            )
                    else:
                        detection_miss_count = 0
                if frame_index % pose_stride == 0:
                    last_pose = pose_estimator.estimate(resized)
                feedback: List[FeedbackMessage] = []
                focus_label: Optional[str] = None
                if current_detection and last_pose is not None:
                    det_label = current_detection.label
                    if det_label in coach.thresholds:
                        exercise_state, detection_feedback = coach.update(det_label, last_pose)
                        feedback.extend(detection_feedback)
                        focus_label = det_label
                        summary.exercise = det_label if config.exercise == "all" else config.exercise
                if focus_label is None and exercise_state.name in coach.thresholds:
                    focus_label = exercise_state.name
                if focus_label and last_pose is not None:
                    if focus_label not in joint_tip_announced:
                        tip_text = JOINT_FOCUS_TIPS.get(focus_label)
                        if tip_text:
                            feedback.append(FeedbackMessage(tip_text, "info"))
                        joint_tip_announced.add(focus_label)
                    visibility_feedback = self._evaluate_pose_visibility(focus_label, last_pose)
                    now_prompt = time.perf_counter()
                    if (
                        visibility_feedback is not None
                        and now_prompt - last_visibility_prompt >= runtime_cfg.audio.get("warning_cooldown_seconds", 4)
                    ):
                        feedback.append(visibility_feedback)
                        last_visibility_prompt = now_prompt
                frame_warnings = sum(1 for fb in feedback if fb.severity in {"warning", "critical"})
                warnings_total += frame_warnings
                highlight_label = None
                if current_detection and current_detection.label in PRIMARY_INDEXES:
                    highlight_label = current_detection.label
                elif exercise_state.name in PRIMARY_INDEXES:
                    highlight_label = exercise_state.name
                annotated = overlay.draw(
                    resized,
                    current_detection,
                    exercise_state,
                    feedback,
                    pose=last_pose,
                    show_metrics=config.show_metrics,
                    show_skeleton=config.show_skeleton,
                    switch_prompt=None,
                    highlight_label=highlight_label,
                )
                fps_meter.tick()
                metrics_clean: Dict[str, object] = {}
                for key, value in exercise_state.metrics.items():
                    try:
                        metrics_clean[key] = float(value)
                    except (TypeError, ValueError):
                        metrics_clean[key] = value
                encoded = self._encode_frame(annotated)
                payload = {
                    "timestamp": time.time(),
                    "frame": encoded,
                    "fps": fps_meter.get_fps(),
                    "running": True,
                    "state": {
                        "name": exercise_state.name,
                        "phase": exercise_state.phase,
                        "rep_count": exercise_state.rep_count,
                        "metrics": metrics_clean,
                        "level": exercise_state.level,
                    },
                    "feedback": [self._serialize_feedback(msg) for msg in feedback],
                    "camera": camera_index,
                    "exercise": summary.exercise,
                    "goalReps": config.goal_reps,
                }
                self._latest_payload = payload
                self._status.update(
                    {
                        "running": True,
                        "message": "session running",
                        "exercise": summary.exercise,
                        "level": config.level,
                        "camera": camera_index,
                        "repCount": exercise_state.rep_count,
                    }
                )
                queue = self._frame_queue
                if queue is not None:
                    try:
                        asyncio.run_coroutine_threadsafe(self._enqueue_frame(queue, payload), loop)
                    except RuntimeError:
                        logger.warning("Unable to push frame payload; consumer likely disconnected")
                summary.rep_count = max(summary.rep_count, exercise_state.rep_count)
                summary.warnings = warnings_total
                summary.ended_at = dt.datetime.utcnow()
                frame_index += 1
            logger.info(
                "Session loop completed for user {} | reps={} warnings={}",
                user_id,
                summary.rep_count if summary else 0,
                warnings_total,
            )
        except Exception as exc:
            logger.exception("Session loop error: {}", exc)
            self._status = {"running": False, "message": f"error: {exc}"}
        finally:
            if cap is not None:
                cap.release()
            if pose_estimator is not None:
                pose_estimator.close()
            if detector is not None:
                try:
                    detector.close()
                except AttributeError:
                    pass
            stop_requested = self._stop_event.is_set()
            self._stop_event.clear()
            self._running_user = None
            self._thread = None
            terminal_status: Dict[str, object]
            if self._status.get("message", "").startswith("error"):
                terminal_status = dict(self._status)
                terminal_status["running"] = False
            else:
                terminal_status = {
                    "running": False,
                    "message": "session stopped" if stop_requested else "session finished",
                }
            if summary is not None:
                terminal_status.setdefault("repCount", summary.rep_count)
                terminal_status.setdefault("exercise", summary.exercise)
                terminal_status.setdefault("level", config.level)
            terminal_status.setdefault("goalReps", config.goal_reps)
            if resolved_camera_index is not None:
                terminal_status.setdefault("camera", resolved_camera_index)
            self._status = terminal_status
            final_payload = dict(terminal_status)
            queue = self._frame_queue
            if queue is not None:
                try:
                    asyncio.run_coroutine_threadsafe(self._enqueue_frame(queue, final_payload), loop)
                except RuntimeError:
                    pass
            if summary and summary.rep_count > 0:
                try:
                    record_session(
                        user_id=summary.user_id,
                        exercise=summary.exercise,
                        rep_count=summary.rep_count,
                        warnings=summary.warnings,
                        started_at=summary.started_at,
                        ended_at=summary.ended_at,
                        session_metadata=summary.session_metadata,
                    )
                except Exception as exc:  # pragma: no cover - safety net
                    logger.exception("Failed to persist session summary: {}", exc)
            self._frame_queue = None
            self._summary = summary
            if "message" not in self._status:
                self._status["message"] = "session finished"

    @staticmethod
    async def _enqueue_frame(queue: asyncio.Queue[Dict[str, object]], payload: Dict[str, object]) -> None:
        try:
            if queue.full():
                queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        await queue.put(payload)

    @staticmethod
    def _configure_capture(cap: cv2.VideoCapture, frame_cfg: Dict[str, object]) -> None:
        target_width = int(frame_cfg.get("target_width", 960))
        target_height = int(frame_cfg.get("target_height", int(target_width * 0.75)))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        cap.set(cv2.CAP_PROP_FPS, 30)

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> str:
        success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            raise RuntimeError("failed to encode frame")
        return base64.b64encode(buffer.tobytes()).decode("ascii")

    @staticmethod
    def _select_detection(detections: List[Detection], interest_labels: List[str]) -> Optional[Detection]:
        if not detections:
            return None
        sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        for det in sorted_detections:
            if det.label in interest_labels:
                return det
        return sorted_detections[0]

    @staticmethod
    def _resolve_device(use_gpu_pref: str | bool) -> str:
        if str(use_gpu_pref).lower() in {"cuda", "gpu", "true", "1"}:
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
        return "cpu"

    @staticmethod
    def _evaluate_pose_visibility(exercise_label: str, pose: PoseResult) -> Optional[FeedbackMessage]:
        indexes = PRIMARY_INDEXES.get(exercise_label)
        if not indexes:
            return None
        total = len(indexes)
        if total == 0:
            return None
        visibility_scores = [pose.landmarks[idx][3] for idx in indexes]
        visible = sum(1 for score in visibility_scores if score >= POSE_VISIBILITY_THRESHOLD)
        ratio = visible / total
        if ratio >= 0.85:
            return None
        severity = "warning" if ratio >= 0.6 else "critical"
        coverage_pct = int(ratio * 100)
        if severity == "critical":
            message = f"Too many key joints are missing ({coverage_pct}% visible); reposition yourself or the camera."
        else:
            message = f"Some key joints are partially occluded ({coverage_pct}% visible); tweak your angle."
        hints: List[str] = []
        joint_tip = JOINT_FOCUS_TIPS.get(exercise_label)
        if joint_tip:
            hints.append(joint_tip)
        return FeedbackMessage(message, severity, hints or None)

    @staticmethod
    def _serialize_feedback(message: FeedbackMessage) -> Dict[str, object]:
        return {
            "message": message.message,
            "severity": message.severity,
            "hints": message.hints or [],
        }


session_manager = ExerciseSession()
