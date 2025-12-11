from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ConfigSelection:
    exercise: str
    level: str
    camera_index: int
    show_skeleton: bool
    show_metrics: bool
    smart_switch: bool
    log_stats: bool
    enable_tts: bool
    enable_beep: bool


class ConfigCancelledError(RuntimeError):
    pass


def prompt_user_config(
    default_exercise: str,
    default_level: str,
    available_cameras: List[int],
    show_skeleton_default: bool,
    show_metrics_default: bool,
    smart_switch_default: bool,
    log_stats_default: bool,
    enable_tts_default: bool,
    enable_beep_default: bool,
) -> ConfigSelection:
    if not available_cameras:
        raise RuntimeError("No cameras detected. Please connect a webcam and retry.")

    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Tkinter is required for the configuration panel but is not available in this environment."
        ) from exc

    exercise_options = ["pushup", "pullup", "squat", "all"]
    level_options = ["beginner", "intermediate", "advanced"]

    if default_exercise not in exercise_options:
        default_exercise = "all"
    if default_level not in level_options:
        default_level = "beginner"

    root = tk.Tk()
    root.title("Form Coach Setup")
    root.resizable(False, False)

    container = ttk.Frame(root, padding=16)
    container.grid(column=0, row=0, sticky="nsew")

    exercise_var = tk.StringVar(value=default_exercise)
    level_var = tk.StringVar(value=default_level)
    skeleton_var = tk.BooleanVar(value=show_skeleton_default)
    metrics_var = tk.BooleanVar(value=show_metrics_default)
    smart_switch_var = tk.BooleanVar(value=smart_switch_default)
    log_stats_var = tk.BooleanVar(value=log_stats_default)
    tts_var = tk.BooleanVar(value=enable_tts_default)
    beep_var = tk.BooleanVar(value=enable_beep_default)

    ttk.Label(container, text="Exercise Mode:").grid(column=0, row=0, sticky="w")
    exercise_box = ttk.Combobox(container, textvariable=exercise_var, values=exercise_options, state="readonly")
    exercise_box.grid(column=0, row=1, sticky="ew", pady=(0, 12))

    ttk.Label(container, text="User Level:").grid(column=0, row=2, sticky="w")
    level_box = ttk.Combobox(container, textvariable=level_var, values=level_options, state="readonly")
    level_box.grid(column=0, row=3, sticky="ew", pady=(0, 12))

    ttk.Label(container, text="Camera:").grid(column=0, row=4, sticky="w")
    camera_labels = [f"Camera {idx}" for idx in available_cameras]
    camera_map = dict(zip(camera_labels, available_cameras))
    camera_box = ttk.Combobox(container, values=camera_labels, state="readonly")
    camera_box.grid(column=0, row=5, sticky="ew", pady=(0, 12))
    camera_box.current(0)

    toggles = ttk.LabelFrame(container, text="Display & Behavior", padding=(12, 8))
    toggles.grid(column=0, row=6, sticky="ew", pady=(0, 12))
    ttk.Checkbutton(toggles, text="Show pose skeleton", variable=skeleton_var).grid(column=0, row=0, sticky="w")
    ttk.Checkbutton(toggles, text="Show metrics & angles", variable=metrics_var).grid(column=0, row=1, sticky="w")
    ttk.Checkbutton(toggles, text="Offer smart exercise switch", variable=smart_switch_var).grid(column=0, row=2, sticky="w")
    ttk.Checkbutton(toggles, text="Log FPS & latency every 60 frames", variable=log_stats_var).grid(column=0, row=3, sticky="w")

    audio_frame = ttk.LabelFrame(container, text="Audio Feedback", padding=(12, 8))
    audio_frame.grid(column=0, row=7, sticky="ew", pady=(0, 12))
    ttk.Checkbutton(audio_frame, text="Enable voice coaching", variable=tts_var).grid(column=0, row=0, sticky="w")
    ttk.Checkbutton(audio_frame, text="Enable rep beep", variable=beep_var).grid(column=0, row=1, sticky="w")

    selection: dict[str, object] = {}

    def on_start() -> None:
        camera_label = camera_box.get()
        selection["exercise"] = exercise_var.get()
        selection["level"] = level_var.get()
        selection["camera_index"] = camera_map[camera_label]
        selection["show_skeleton"] = skeleton_var.get()
        selection["show_metrics"] = metrics_var.get()
        selection["smart_switch"] = smart_switch_var.get()
        selection["log_stats"] = log_stats_var.get()
        selection["enable_tts"] = tts_var.get()
        selection["enable_beep"] = beep_var.get()
        root.destroy()

    def on_cancel() -> None:
        selection.clear()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_cancel)

    start_btn = ttk.Button(container, text="Start", command=on_start)
    start_btn.grid(column=0, row=8, sticky="ew", pady=(4, 0))
    root.mainloop()

    if not selection:
        raise ConfigCancelledError("Configuration panel was closed before starting.")

    return ConfigSelection(
        exercise=str(selection["exercise"]),
        level=str(selection["level"]),
        camera_index=int(selection["camera_index"]),
        show_skeleton=bool(selection["show_skeleton"]),
        show_metrics=bool(selection["show_metrics"]),
        smart_switch=bool(selection["smart_switch"]),
        log_stats=bool(selection["log_stats"]),
        enable_tts=bool(selection["enable_tts"]),
        enable_beep=bool(selection["enable_beep"]),
    )
