import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st

from src.detectors.yolo_detector import YOLOExerciseDetector
from src.logic.rules import RuleBasedCoach
from src.pose.mediapipe_pose import MediaPipePoseEstimator
from src.ui.overlay import FrameOverlay, JOINT_FOCUS_TIPS, PRIMARY_INDEXES
from src.ui.audio_feedback import AudioFeedbackManager
from src.utils.camera import enumerate_cameras
from src.utils.config import RuntimeConfig, load_runtime_config
from src.utils.profiler import FPSMeter
from src.utils.structures import Detection, ExerciseState, FeedbackMessage, PoseResult


FIXED_WEIGHTS_PATH = Path("weights/yolov8n-exercise.pt")


def _resolve_device(use_gpu_pref: str) -> str:
    if use_gpu_pref == "auto":
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return "cuda" if str(use_gpu_pref).lower() in {"true", "1", "cuda", "gpu"} else "cpu"


def _configure_capture(cap: cv2.VideoCapture, frame_cfg: dict) -> None:
    target_width = int(frame_cfg.get("target_width", 960))
    target_height = int(frame_cfg.get("target_height", int(target_width * 0.75)))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    cap.set(cv2.CAP_PROP_FPS, 30)


def app() -> None:
    st.set_page_config(page_title="Exercise Coach", layout="wide")
    st.title("Real-time Exercise Form Coach")

    if not FIXED_WEIGHTS_PATH.is_file():
        st.error(f"Missing weights file at {FIXED_WEIGHTS_PATH}. Upload or place it and rerun.")
        return

    # Sidebar: configuration
    st.sidebar.header("Configuration")
    runtime_path = st.sidebar.text_input("Runtime config path", "configs/runtime.yaml")
    levels_path = st.sidebar.text_input("Exercise levels config", "configs/exercise_levels.yaml")
    runtime_cfg: RuntimeConfig = load_runtime_config(runtime_path)

    available_cameras: List[int] = enumerate_cameras()
    camera_index = st.sidebar.selectbox("Camera index", available_cameras or [0], index=0)

    exercises = ["all", "squat", "pullup", "pushup"]
    exercise_choice = st.sidebar.selectbox("Exercise focus", exercises, index=0)
    level_choice = st.sidebar.selectbox("Coaching level", ["beginner", "intermediate", "advanced"], index=0)

    show_skeleton = bool(runtime_cfg.display.get("show_skeleton", True))
    show_metrics = bool(runtime_cfg.display.get("show_metrics", True))
    smart_switch_enabled = bool(runtime_cfg.display.get("allow_smart_switch", True))
    log_stats = bool(runtime_cfg.display.get("log_stats", False))
    audio_cfg = runtime_cfg.audio if runtime_cfg.audio else {}
    enable_tts_default = bool(audio_cfg.get("enable_tts", True))
    enable_beep_default = bool(audio_cfg.get("enable_beep", True))
    voice_rate = int(audio_cfg.get("voice_rate", 175))
    warning_cooldown = float(audio_cfg.get("warning_cooldown_seconds", 4))
    beep_volume = float(audio_cfg.get("beep_volume", 0.8))
    announce_reps_tts = bool(audio_cfg.get("announce_reps_tts", False))

    show_skeleton = st.sidebar.checkbox("Show skeleton", value=show_skeleton)
    show_metrics = st.sidebar.checkbox("Show metrics", value=show_metrics)
    smart_switch_enabled = st.sidebar.checkbox("Smart exercise switch", value=smart_switch_enabled)
    enable_tts = st.sidebar.checkbox("Voice coaching (TTS)", value=enable_tts_default)
    enable_beep = st.sidebar.checkbox("Rep beep", value=enable_beep_default)

    run_button = st.sidebar.button("Start")
    stop_button = st.sidebar.button("Stop")

    frame_slot = st.empty()
    info_slot = st.empty()
    stats_slot = st.empty()

    if "running" not in st.session_state:
        st.session_state.running = False

    if run_button:
        st.session_state.running = True
    if stop_button:
        st.session_state.running = False

    if not st.session_state.running:
        st.info("Click Start to begin processing.")
        return

    # Initialize pipeline components
    device = _resolve_device(runtime_cfg.latency.get("use_gpu", "auto"))
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
    overlay = FrameOverlay(
        alpha=float(runtime_cfg.latency.get("overlay_alpha", 0.75)),
        font_scale=float(runtime_cfg.display.get("font_scale", 0.6)),
        metric_font_scale=float(runtime_cfg.display.get("metric_font_scale", 0.5)),
        margin=int(runtime_cfg.display.get("hud_margin", 16)),
        primary_thickness=int(runtime_cfg.display.get("skeleton_primary_thickness", 3)),
        secondary_thickness=int(runtime_cfg.display.get("skeleton_secondary_thickness", 1)),
    )
    coach = RuleBasedCoach(
        levels_config_path=str(levels_path),
        level=level_choice,
        smoothing_window=int(runtime_cfg.latency.get("smoothing_window", 5)),
    )
    all_exercises = list(coach.thresholds.keys())
    default_state_name = exercise_choice if exercise_choice != "all" else "idle"

    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error(f"Failed to open camera index {camera_index}. Available: {available_cameras}")
        pose_estimator.close()
        return
    _configure_capture(cap, runtime_cfg.frame)

    fps = FPSMeter()
    audio_manager: Optional[AudioFeedbackManager] = None
    if enable_tts or enable_beep:
        audio_manager = AudioFeedbackManager(
            enable_tts=enable_tts,
            enable_beep=enable_beep,
            voice_rate=voice_rate,
            beep_volume=beep_volume,
        )
    current_detection: Optional[Detection] = None
    current_state = ExerciseState(name=default_state_name, phase="idle", rep_count=0, metrics={}, level=level_choice)
    last_pose_result: Optional[PoseResult] = None
    detection_stride = int(runtime_cfg.frame.get("detection_stride", 3))
    pose_stride = int(runtime_cfg.frame.get("pose_stride", 1))
    frame_index = 0
    target_width = int(runtime_cfg.frame.get("target_width", 960))

    last_rep_count = 0
    last_voice_time = 0.0
    last_voice_message: Optional[str] = None

    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera read failed; stopping.")
                break

            if runtime_cfg.frame.get("flip", False):
                frame = cv2.flip(frame, 1)
            if runtime_cfg.frame.get("mirror", False):
                frame = cv2.flip(frame, 0)

            scale = target_width / frame.shape[1]
            target_height = int(frame.shape[0] * scale)
            resized = cv2.resize(frame, (target_width, target_height))

            detections: List[Detection] = []
            if frame_index % detection_stride == 0:
                try:
                    detections = detector.detect(resized)
                except Exception:
                    detections = []

            if detections:
                detections_sorted = sorted(detections, key=lambda d: d.confidence, reverse=True)
                focus = None
                interest_labels = all_exercises[:] if exercise_choice == "all" else [exercise_choice]
                for det in detections_sorted:
                    if det.label in interest_labels:
                        focus = det
                        break
                current_detection = focus or detections_sorted[0]

            if frame_index % pose_stride == 0:
                try:
                    last_pose_result = pose_estimator.estimate(resized)
                except Exception:
                    last_pose_result = None

            feedback_messages: List[FeedbackMessage] = []
            if current_detection is not None and last_pose_result is not None:
                det_label = current_detection.label
                if det_label in coach.thresholds:
                    state, detection_feedback = coach.update(det_label, last_pose_result)
                    current_state = state
                    feedback_messages.extend(detection_feedback)
                else:
                    current_state = ExerciseState(
                        name=default_state_name,
                        phase="idle",
                        rep_count=current_state.rep_count,
                        metrics={},
                        level=level_choice,
                    )

            # Audio feedback for reps and warnings
            if audio_manager is not None:
                if current_state.rep_count > last_rep_count:
                    if announce_reps_tts and enable_tts:
                        if current_state.name and current_state.name != "idle":
                            audio_manager.enqueue_tts(f"{current_state.name} {current_state.rep_count}")
                        else:
                            audio_manager.enqueue_tts(f"{current_state.rep_count}")
                    if enable_beep:
                        audio_manager.enqueue_beep("rep")
                last_rep_count = current_state.rep_count

                voice_candidate = next((fb for fb in feedback_messages if fb.severity in {"critical", "warning"}), None)
                if voice_candidate:
                    now = time.perf_counter()
                    if now - last_voice_time >= warning_cooldown:
                        audio_manager.enqueue_tts(voice_candidate.message, voice_candidate.severity)
                        if voice_candidate.severity == "critical":
                            audio_manager.enqueue_beep("warning")
                        last_voice_time = now
                        last_voice_message = voice_candidate.message

            fps.tick()
            highlight_label = None
            if current_detection and current_detection.label in PRIMARY_INDEXES:
                highlight_label = current_detection.label
            elif current_state.name in PRIMARY_INDEXES:
                highlight_label = current_state.name

            annotated = overlay.draw(
                resized,
                current_detection,
                current_state,
                feedback_messages,
                pose=last_pose_result,
                show_metrics=show_metrics,
                show_skeleton=show_skeleton,
                switch_prompt=None,
                highlight_label=highlight_label,
            )

            # Convert BGR to RGB for Streamlit display
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_slot.image(rgb, use_column_width=True)
            info_slot.markdown(f"**FPS:** {fps.get_fps():.1f} | **Reps:** {current_state.rep_count} | **Phase:** {current_state.phase}")
            if feedback_messages:
                st.sidebar.write("Latest feedback:")
                for fb in feedback_messages[:5]:
                    st.sidebar.write(f"- ({fb.severity}) {fb.message}")
            if log_stats and frame_index % 120 == 0:
                stats_slot.write(f"Stride d/p {detection_stride}/{pose_stride}")

            frame_index += 1
            # Yield control to Streamlit to keep UI responsive
            time.sleep(0.001)
    finally:
        cap.release()
        pose_estimator.close()
        if audio_manager is not None:
            audio_manager.stop()


if __name__ == "__main__":
    app()
