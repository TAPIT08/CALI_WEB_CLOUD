from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", message=r"SymbolDatabase\.GetPrototype\(\) is deprecated", category=UserWarning)
try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:  # pragma: no cover
    pass

import cv2
import numpy as np

from src.detectors.yolo_detector import YOLOExerciseDetector
from src.logic.rules import RuleBasedCoach
from src.pose.mediapipe_pose import MediaPipePoseEstimator
from src.ui.audio_feedback import AudioFeedbackManager
from src.ui.config_panel import ConfigCancelledError, prompt_user_config
from src.ui.overlay import FrameOverlay, JOINT_FOCUS_TIPS, PRIMARY_INDEXES
from src.utils.camera import enumerate_cameras
from src.utils.config import RuntimeConfig, load_runtime_config
from src.utils.profiler import FPSMeter
from src.utils.structures import Detection, ExerciseState, FeedbackMessage, PoseResult


FIXED_WEIGHTS_PATH = Path("weights/yolov8n-exercise.pt")
STANDING_LABELS: Set[str] = {"standing", "stand"}
CAMERA_GUIDANCE: Dict[str, str] = {
    "pushup": "Angle the camera about 30-45 degrees off your side so shoulders through ankles stay in frame.",
    "squat": "Keep the camera slightly off-center or side-on to capture hips, knees, and ankles together.",
    "pullup": "Aim for a slight front-side angle so elbows, shoulders, and hips stay visible the whole rep.",
}
CAMERA_PROMPT_COOLDOWN = 3.0
POSE_VISIBILITY_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time exercise form checker")
    parser.add_argument("--exercise", type=str, default="all", choices=["pushup", "pullup", "squat", "all"], help="Exercise focus")
    parser.add_argument("--level", type=str, default="beginner", choices=["beginner", "intermediate", "advanced"], help="Coaching level")
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.yaml"), help="Runtime configuration")
    parser.add_argument("--levels-config", type=Path, default=Path("configs/exercise_levels.yaml"), help="Exercise level thresholds")
    parser.add_argument("--camera", type=int, default=-1, help="Preferred camera index (-1 for auto)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device preference")
    parser.add_argument("--headless", action="store_true", help="Disable window display for benchmarking")
    parser.add_argument("--detection-stride", type=int, default=None, help="Override detection stride from runtime config")
    parser.add_argument("--pose-stride", type=int, default=None, help="Override pose stride from runtime config")
    return parser.parse_args()


def resolve_device(device_arg: str, use_gpu_pref: str) -> str:
    if device_arg != "auto":
        return "cuda" if device_arg == "cuda" else "cpu"
    if use_gpu_pref == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            return "cpu"
        return "cpu"
    if use_gpu_pref and str(use_gpu_pref).lower() in {"true", "1", "cuda", "gpu"}:
        return "cuda"
    return "cpu"


def configure_capture(cap: cv2.VideoCapture, frame_cfg: dict) -> None:
    target_width = frame_cfg.get("target_width", 960)
    target_height = frame_cfg.get("target_height", int(target_width * 0.75))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    cap.set(cv2.CAP_PROP_FPS, 30)


def evaluate_pose_visibility(exercise_label: str, pose: PoseResult) -> Optional[FeedbackMessage]:
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
    hints = []
    joint_tip = JOINT_FOCUS_TIPS.get(exercise_label)
    if joint_tip:
        hints.append(joint_tip)
    guidance_tip = CAMERA_GUIDANCE.get(exercise_label)
    if guidance_tip:
        hints.append(guidance_tip)
    return FeedbackMessage(message, severity, hints if hints else None)


def main() -> None:
    args = parse_args()
    runtime_cfg: RuntimeConfig = load_runtime_config(str(args.runtime_config))
    if not FIXED_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Expected YOLO weights at {FIXED_WEIGHTS_PATH}. Please place the file before running.")
    weights_path = FIXED_WEIGHTS_PATH
    display_cfg = runtime_cfg.display if runtime_cfg.display else {}
    show_metrics_default = bool(display_cfg.get("show_metrics", True))
    show_skeleton_default = bool(display_cfg.get("show_skeleton", True))
    smart_switch_default = bool(display_cfg.get("allow_smart_switch", True))
    log_stats_default = bool(display_cfg.get("log_stats", False))
    font_scale = float(display_cfg.get("font_scale", 0.6))
    metric_font_scale = float(display_cfg.get("metric_font_scale", 0.5))
    hud_margin = int(display_cfg.get("hud_margin", 16))
    primary_thickness = int(display_cfg.get("skeleton_primary_thickness", 3))
    secondary_thickness = int(display_cfg.get("skeleton_secondary_thickness", 1))

    audio_cfg = runtime_cfg.audio if runtime_cfg.audio else {}
    enable_tts_default = bool(audio_cfg.get("enable_tts", True))
    enable_beep_default = bool(audio_cfg.get("enable_beep", True))
    voice_rate = int(audio_cfg.get("voice_rate", 175))
    warning_cooldown = float(audio_cfg.get("warning_cooldown_seconds", 4))
    beep_volume = float(audio_cfg.get("beep_volume", 0.8))
    announce_reps_tts = bool(audio_cfg.get("announce_reps_tts", False))

    available_cameras: List[int] = enumerate_cameras()
    auto_camera = available_cameras[0] if available_cameras else 0

    if args.headless:
        exercise_choice = args.exercise
        level_choice = args.level
        camera_index = args.camera if args.camera >= 0 else auto_camera
        if available_cameras and camera_index not in available_cameras:
            camera_index = available_cameras[0]
        show_skeleton = show_skeleton_default
        show_metrics = show_metrics_default
        smart_switch_enabled = smart_switch_default
        log_stats = log_stats_default
        enable_tts = enable_tts_default
        enable_beep = enable_beep_default
    else:
        try:
            selection = prompt_user_config(
                args.exercise,
                args.level,
                available_cameras or [auto_camera],
                show_skeleton_default,
                show_metrics_default,
                smart_switch_default,
                log_stats_default,
                enable_tts_default,
                enable_beep_default,
            )
        except ConfigCancelledError:
            print("Configuration cancelled by user. Exiting.")
            return
        exercise_choice = selection.exercise
        level_choice = selection.level
        camera_index = selection.camera_index
        show_skeleton = selection.show_skeleton
        show_metrics = selection.show_metrics
        smart_switch_enabled = selection.smart_switch
        log_stats = selection.log_stats
        enable_tts = selection.enable_tts
        enable_beep = selection.enable_beep

    if camera_index < 0:
        camera_index = auto_camera

    device = resolve_device(args.device, runtime_cfg.latency.get("use_gpu", "auto"))
    try:
        from ultralytics.utils import LOGGER

        LOGGER.setLevel(logging.ERROR)
    except Exception:  # pragma: no cover
        pass
    detector = YOLOExerciseDetector(
        weights_path=str(weights_path),
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
        levels_config_path=str(args.levels_config),
        level=level_choice,
        smoothing_window=int(runtime_cfg.latency.get("smoothing_window", 5)),
    )
    all_exercises = list(coach.thresholds.keys())
    overlay = FrameOverlay(
        alpha=float(runtime_cfg.latency.get("overlay_alpha", 0.75)),
        font_scale=font_scale,
        metric_font_scale=metric_font_scale,
        margin=hud_margin,
        primary_thickness=primary_thickness,
        secondary_thickness=secondary_thickness,
    )
    fps_meter = FPSMeter()
    audio_manager: Optional[AudioFeedbackManager] = None
    if enable_tts or enable_beep:
        audio_manager = AudioFeedbackManager(
            enable_tts=enable_tts,
            enable_beep=enable_beep,
            voice_rate=voice_rate,
            beep_volume=beep_volume,
        )
    detector_executor = ThreadPoolExecutor(max_workers=1)
    pending_detection: Optional[Future[tuple[List[Detection], float]]] = None

    default_state_name = exercise_choice if exercise_choice != "all" else "idle"
    interest_labels = all_exercises[:] if exercise_choice == "all" else [exercise_choice]
    joint_tip_announced: Set[str] = set()
    last_camera_prompt_time = 0.0

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        available_msg = f" Available cameras: {available_cameras}" if available_cameras else ""
        raise RuntimeError(f"Failed to open camera index {camera_index}.{available_msg}")
    configure_capture(cap, runtime_cfg.frame)

    detection_stride = int(runtime_cfg.frame.get("detection_stride", 3))
    pose_stride = int(runtime_cfg.frame.get("pose_stride", 1))
    if args.detection_stride and args.detection_stride > 0:
        detection_stride = max(1, args.detection_stride)
    if args.pose_stride and args.pose_stride > 0:
        pose_stride = max(1, args.pose_stride)

    frame_index = 0
    target_width = int(runtime_cfg.frame.get("target_width", 960))
    current_detection: Optional[Detection] = None
    detection_miss_count = 0
    current_state = ExerciseState(name=default_state_name, phase="idle", rep_count=0, metrics={}, level=level_choice)
    current_feedback: List[FeedbackMessage] = []
    last_pose_result: Optional[PoseResult] = None
    pending_switch_label: Optional[str] = None
    detection_durations: List[float] = []
    pose_durations: List[float] = []
    last_rep_count = 0
    last_voice_time = 0.0
    last_voice_message: Optional[str] = None

    def run_detection(frame_input: np.ndarray) -> tuple[List[Detection], float]:
        start = time.perf_counter()
        detections = detector.detect(frame_input)
        return detections, time.perf_counter() - start

    def prioritize_exercise(preferred: str) -> None:
        nonlocal interest_labels, exercise_choice, default_state_name, pending_switch_label
        nonlocal current_detection, detection_miss_count, current_state, last_rep_count, joint_tip_announced
        if preferred not in coach.thresholds:
            return
        joint_tip_announced.discard(preferred)
        if exercise_choice == "all":
            remaining = [label for label in all_exercises if label != preferred]
            interest_labels = [preferred] + remaining
            print(f"Prioritizing {preferred} detections while in 'all' mode.")
            return
        if exercise_choice == preferred:
            return
        interest_labels = [preferred]
        exercise_choice = preferred
        default_state_name = preferred
        pending_switch_label = None
        current_detection = None
        detection_miss_count = detection_stride * 2
        current_state = ExerciseState(
            name=default_state_name,
            phase="idle",
            rep_count=0,
            metrics={},
            level=level_choice,
        )
        last_rep_count = 0
        print(f"Switched focus to {preferred}.")

    def process_detections(detections: List[Detection], det_duration: float) -> None:
        nonlocal current_detection, detection_miss_count, pending_switch_label
        if log_stats:
            detection_durations.append(det_duration)
        detections_sorted = sorted(detections, key=lambda d: d.confidence, reverse=True)
        focus_detection: Optional[Detection] = None
        for det in detections_sorted:
            if det.label in interest_labels:
                focus_detection = det
                break
        if focus_detection is None:
            for det in detections_sorted:
                if det.label in STANDING_LABELS:
                    focus_detection = det
                    break
        if focus_detection is None and exercise_choice == "all" and detections_sorted:
            focus_detection = detections_sorted[0]

        if focus_detection:
            current_detection = focus_detection
            detection_miss_count = 0
        else:
            detection_miss_count += 1
            if detection_miss_count >= detection_stride * 2:
                current_detection = None

        if not smart_switch_enabled or exercise_choice == "all":
            return

        top_detection = detections_sorted[0] if detections_sorted else None
        conf_thresh = runtime_cfg.yolo.get("conf_threshold", 0.35)
        if (
            top_detection
            and top_detection.confidence >= conf_thresh
            and top_detection.label != interest_labels[0]
            and top_detection.label not in STANDING_LABELS
        ):
            pending_switch_label = top_detection.label
        elif top_detection and top_detection.label == interest_labels[0]:
            pending_switch_label = None
        elif top_detection and top_detection.label in STANDING_LABELS:
            pending_switch_label = None
        elif not detections_sorted:
            pending_switch_label = None

    try:
        while True:
            switch_prompt: Optional[str] = None
            ret, frame = cap.read()
            if not ret:
                break
            if runtime_cfg.frame.get("flip", False):
                frame = cv2.flip(frame, 1)
            if runtime_cfg.frame.get("mirror", False):
                frame = cv2.flip(frame, 0)

            scale = target_width / frame.shape[1]
            target_height = int(frame.shape[0] * scale)
            resized = cv2.resize(frame, (target_width, target_height))

            if pending_detection is not None and pending_detection.done():
                try:
                    detection_result = pending_detection.result()
                    detections, det_duration = detection_result
                except Exception:
                    detections, det_duration = [], 0.0
                pending_detection = None
                process_detections(detections, det_duration)

            if frame_index % detection_stride == 0 and pending_detection is None:
                pending_detection = detector_executor.submit(run_detection, resized.copy())

            if frame_index % pose_stride == 0:
                pose_start = time.perf_counter()
                pose_candidate = pose_estimator.estimate(resized)
                pose_duration = time.perf_counter() - pose_start
                if log_stats:
                    pose_durations.append(pose_duration)
                if pose_candidate is not None:
                    last_pose_result = pose_candidate

            feedback_messages: List[FeedbackMessage] = []
            focus_label_for_visibility: Optional[str] = None

            if current_detection is not None and last_pose_result is not None:
                det_label = current_detection.label
                if det_label in STANDING_LABELS:
                    feedback_messages.append(
                        FeedbackMessage(
                            "Standing detected. Choose the next focus exercise.",
                            "info",
                            [
                                "Press 1 to prioritize squat feedback.",
                                "Press 2 to prioritize pullup feedback.",
                            ],
                        )
                    )
                    prev_rep = current_state.rep_count
                    current_state = ExerciseState(
                        name=default_state_name,
                        phase="idle",
                        rep_count=prev_rep,
                        metrics={},
                        level=level_choice,
                    )
                elif det_label in coach.thresholds:
                    state, detection_feedback = coach.update(det_label, last_pose_result)
                    current_state = state
                    feedback_messages.extend(detection_feedback)
                    focus_label_for_visibility = det_label
                else:
                    prev_rep = current_state.rep_count
                    current_state = ExerciseState(
                        name=default_state_name,
                        phase="idle",
                        rep_count=prev_rep,
                        metrics={},
                        level=level_choice,
                    )
            elif current_detection is None:
                prev_rep = current_state.rep_count
                current_state = ExerciseState(
                    name=default_state_name,
                    phase="idle",
                    rep_count=prev_rep,
                    metrics={},
                    level=level_choice,
                )

            if focus_label_for_visibility is None:
                if current_state.name in coach.thresholds:
                    focus_label_for_visibility = current_state.name
                elif exercise_choice != "all" and exercise_choice in coach.thresholds:
                    focus_label_for_visibility = exercise_choice

            if focus_label_for_visibility and last_pose_result is not None:
                if focus_label_for_visibility not in joint_tip_announced:
                    tip_text = JOINT_FOCUS_TIPS.get(focus_label_for_visibility)
                    if tip_text:
                        feedback_messages.append(FeedbackMessage(tip_text, "info"))
                    joint_tip_announced.add(focus_label_for_visibility)
                visibility_feedback = evaluate_pose_visibility(focus_label_for_visibility, last_pose_result)
                if visibility_feedback is not None:
                    now_prompt = time.perf_counter()
                    if now_prompt - last_camera_prompt_time >= CAMERA_PROMPT_COOLDOWN:
                        feedback_messages.append(visibility_feedback)
                        last_camera_prompt_time = now_prompt

            current_feedback = feedback_messages

            if audio_manager is not None:
                        if current_state.rep_count > last_rep_count:
                            # On rep increment: optionally announce current reps per exercise
                            if announce_reps_tts and enable_tts:
                                # Speak either the exercise name with rep count or just the number
                                if current_state.name and current_state.name != "idle":
                                    audio_manager.enqueue_tts(f"{current_state.name} {current_state.rep_count}")
                                else:
                                    audio_manager.enqueue_tts(f"{current_state.rep_count}")
                            if enable_beep:
                                audio_manager.enqueue_beep("rep")
                last_rep_count = current_state.rep_count

                voice_candidate = next(
                    (fb for fb in current_feedback if fb.severity in {"critical", "warning"}),
                    None,
                )
                if voice_candidate:
                    now = time.perf_counter()
                    if now - last_voice_time >= warning_cooldown:
                        audio_manager.enqueue_tts(voice_candidate.message, voice_candidate.severity)
                        if voice_candidate.severity == "critical":
                            audio_manager.enqueue_beep("warning")
                        last_voice_time = now
                        last_voice_message = voice_candidate.message

            fps_meter.tick()

            if smart_switch_enabled and pending_switch_label and exercise_choice != "all":
                switch_prompt = f"Detected {pending_switch_label}. Press 's' to switch."

            highlight_label = None
            if current_detection and current_detection.label in PRIMARY_INDEXES:
                highlight_label = current_detection.label
            elif current_state.name in PRIMARY_INDEXES:
                highlight_label = current_state.name

            if not args.headless:
                annotated = overlay.draw(
                    resized,
                    current_detection,
                    current_state,
                    current_feedback,
                    pose=last_pose_result,
                    show_metrics=show_metrics,
                    show_skeleton=show_skeleton,
                    switch_prompt=switch_prompt,
                    highlight_label=highlight_label,
                )
                cv2.putText(
                    annotated,
                    f"FPS: {fps_meter.get_fps():.1f}",
                    (annotated.shape[1] - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
                cv2.imshow("Form Coach", annotated)

            frame_index += 1
            if log_stats and frame_index % 60 == 0:
                avg_det_ms = (sum(detection_durations) / len(detection_durations) * 1000) if detection_durations else 0.0
                avg_pose_ms = (sum(pose_durations) / len(pose_durations) * 1000) if pose_durations else 0.0
                print(
                    f"[Stats] FPS~{fps_meter.get_fps():.1f} | detection {avg_det_ms:.1f} ms | pose {avg_pose_ms:.1f} ms | stride d/p {detection_stride}/{pose_stride}"
                )
                detection_durations.clear()
                pose_durations.clear()

            if not args.headless:
                key = cv2.waitKey(1)
                if key & 0xFF == ord("q"):
                    break
                if key & 0xFF in {ord("1"), ord("2")}:
                    preferred_label = "squat" if key & 0xFF == ord("1") else "pullup"
                    prioritize_exercise(preferred_label)
                    pending_switch_label = None
                if smart_switch_enabled and pending_switch_label and key & 0xFF in {ord("s"), ord("S")}:
                    prioritize_exercise(pending_switch_label)
                    pending_switch_label = None
    finally:
        cap.release()
        pose_estimator.close()
        cv2.destroyAllWindows()
        if pending_detection is not None and not pending_detection.done():
            pending_detection.cancel()
        detector_executor.shutdown(wait=False)
        if audio_manager is not None:
            audio_manager.stop()


if __name__ == "__main__":
    main()
