from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import simpleaudio as sa

try:
    import pyttsx3
except ImportError:  # pragma: no cover
    pyttsx3 = None  # type: ignore[assignment]


@dataclass
class AudioEvent:
    kind: Literal["tts", "beep"]
    payload: str
    urgency: Literal["info", "warning", "critical"] = "info"


class AudioFeedbackManager:
    def __init__(
        self,
        enable_tts: bool,
        enable_beep: bool,
        voice_rate: int,
        beep_volume: float,
    ) -> None:
        self.enable_tts = enable_tts and pyttsx3 is not None
        self.enable_beep = enable_beep
        self.voice_rate = voice_rate
        self.beep_volume = float(max(0.0, min(1.0, beep_volume)))
        self.queue: "queue.Queue[AudioEvent]" = queue.Queue()
        self._stop_event = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()
        self.tts_engine = None
        if self.enable_tts and pyttsx3 is not None:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty("rate", voice_rate)

    def enqueue_tts(self, text: str, severity: Literal["info", "warning", "critical"] = "info") -> None:
        if not text or not self.enable_tts:
            return
        self.queue.put(AudioEvent(kind="tts", payload=text, urgency=severity))

    def enqueue_beep(self, tone: Literal["rep", "warning"] = "rep") -> None:
        if not self.enable_beep:
            return
        freq = 880 if tone == "rep" else 440
        duration = 0.15 if tone == "rep" else 0.25
        payload = f"{freq}:{duration}"
        self.queue.put(AudioEvent(kind="beep", payload=payload))

    def stop(self) -> None:
        self._stop_event.set()
        self.queue.put(AudioEvent(kind="beep", payload="0:0"))
        self.worker.join(timeout=2)
        if self.tts_engine is not None:
            self.tts_engine.stop()  # type: ignore[call-arg]

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if event.kind == "tts":
                self._handle_tts(event)
            elif event.kind == "beep":
                self._handle_beep(event)

    def _handle_tts(self, event: AudioEvent) -> None:
        if self.tts_engine is None:
            return
        self.tts_engine.say(event.payload)
        self.tts_engine.runAndWait()

    def _handle_beep(self, event: AudioEvent) -> None:
        freq, duration = self._parse_beep(event.payload)
        if duration <= 0 or freq <= 0:
            return
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(freq * 2 * np.pi * t)
        audio = tone * (32767 * self.beep_volume)
        audio = audio.astype(np.int16)
        sa.play_buffer(audio, 1, 2, sample_rate)

    @staticmethod
    def _parse_beep(payload: str) -> tuple[int, float]:
        try:
            freq_str, dur_str = payload.split(":")
            return int(freq_str), float(dur_str)
        except (ValueError, TypeError):
            return 0, 0.0
