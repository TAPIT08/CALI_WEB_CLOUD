from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.server.database import (
    complete_reminder,
    get_dashboard,
    get_preferences,
    create_user,
    list_achievements,
    list_reminders,
    list_sessions,
    schedule_reminder,
    update_preferences,
    init_db,
    User,
)
from src.server.logging_utils import configure_logging
from src.server.models.schemas import (
    DashboardResponse,
    MessageResponse,
    PreferencesResponse,
    PreferencesUpdate,
    ReminderCreate,
    ReminderUpdate,
    SessionListResponse,
    SessionStartResponse,
    SessionStatusResponse,
    SessionStopResponse,
    StartSessionRequest,
    UserProfile,
)
from src.server.session import (
    SessionAlreadyRunningError,
    SessionConfig,
    SessionNotRunningError,
    session_manager,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app = FastAPI(title="Exercise Coach Web", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def on_startup() -> None:
    configure_logging()
    init_db()
    _get_demo_user()
    print("App running at: http://127.0.0.1:8000")


@lru_cache(maxsize=1)
def _get_demo_user() -> User:
    # Ensure a single shared user record for the local prototype
    return create_user("Local Athlete")


def _build_preferences_response(user_id: str) -> PreferencesResponse:
    prefs = get_preferences(user_id)
    return PreferencesResponse(
        level=prefs.level,
        focus_exercises=prefs.focus_exercises,
        daily_goal=prefs.daily_goal,
        smart_goal=prefs.smart_goal,
        timezone=prefs.timezone,
        reminder_hour=prefs.reminder_hour,
        allow_notifications=prefs.allow_notifications,
    )


def _build_user_profile(user_dashboard: dict) -> UserProfile:
    return UserProfile(
        id=user_dashboard["id"],
        email=user_dashboard["email"],
        display_name=user_dashboard["display_name"],
        avatar_url=user_dashboard.get("avatar_url"),
        member_since=user_dashboard["member_since"],
        level=user_dashboard["level"],
        experience=user_dashboard["experience"],
        xp_to_next_level=user_dashboard["xp_to_next_level"],
    )


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=500, detail="index.html not found")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/users/preferences", response_model=PreferencesResponse)
async def read_preferences() -> PreferencesResponse:
    user = _get_demo_user()
    return _build_preferences_response(user.id)


@app.put("/api/users/preferences", response_model=PreferencesResponse)
async def update_user_preferences(
    payload: PreferencesUpdate,
) -> PreferencesResponse:
    user = _get_demo_user()
    update_preferences(
        user.id,
        level=payload.level,
        focus_exercises=payload.focus_exercises,
        daily_goal=payload.daily_goal,
        smart_goal=payload.smart_goal,
        timezone=payload.timezone,
        reminder_hour=payload.reminder_hour,
        allow_notifications=payload.allow_notifications,
    )
    return _build_preferences_response(user.id)


@app.get("/api/dashboard", response_model=DashboardResponse)
async def read_dashboard() -> DashboardResponse:
    user = _get_demo_user()
    data = get_dashboard(user.id)
    return DashboardResponse(
        user=_build_user_profile(data["user"]),
        preferences=_build_preferences_response(user.id),
        streaks=data["streaks"],
        progress_timeline=data["progress_timeline"],
        recommendations=data["recommendations"],
        achievements=data["achievements"],
        reminders=data["reminders"],
        suggestions=data["suggestions"],
    )


@app.get("/api/sessions/history", response_model=SessionListResponse)
async def session_history(
    limit: int = 20,
) -> SessionListResponse:
    user = _get_demo_user()
    sessions = list_sessions(user.id, limit)
    return SessionListResponse(sessions=sessions)


@app.get("/api/achievements")
async def achievements() -> list[dict]:
    user = _get_demo_user()
    return list_achievements(user.id)


@app.get("/api/reminders")
async def reminders(
    include_completed: bool = False,
) -> list[dict]:
    user = _get_demo_user()
    return list_reminders(user.id, include_completed)


@app.post("/api/reminders", response_model=MessageResponse)
async def create_reminder(
    payload: ReminderCreate,
) -> MessageResponse:
    user = _get_demo_user()
    schedule_reminder(user.id, payload.message, payload.remind_at)
    return MessageResponse(message="Reminder scheduled")


@app.patch("/api/reminders/{reminder_id}", response_model=MessageResponse)
async def update_reminder(
    reminder_id: int,
    payload: ReminderUpdate,
) -> MessageResponse:
    user = _get_demo_user()
    if payload.completed:
        complete_reminder(reminder_id, user.id)
    return MessageResponse(message="Reminder updated")


@app.post("/api/session/start", response_model=SessionStartResponse)
async def start_session(
    request: StartSessionRequest,
) -> SessionStartResponse:
    user = _get_demo_user()
    config = SessionConfig(
        exercise=request.exercise,
        level=request.level,
        camera_index=request.camera_index,
        detection_stride=request.detection_stride,
        pose_stride=request.pose_stride,
        show_skeleton=request.show_skeleton,
        show_metrics=request.show_metrics,
        goal_reps=request.goal_reps,
    )
    try:
        await session_manager.start(user.id, config)
    except SessionAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return SessionStartResponse(
        status="running",
        user_id=user.id,
        exercise=config.exercise,
        goal_reps=config.goal_reps,
    )


@app.post("/api/session/stop", response_model=SessionStopResponse)
async def stop_session() -> SessionStopResponse:
    user = _get_demo_user()
    try:
        await session_manager.stop(user.id)
    except SessionNotRunningError:
        return SessionStopResponse(status="idle")
    return SessionStopResponse(status="stopped")


@app.get("/api/session/status", response_model=SessionStatusResponse)
async def session_status() -> SessionStatusResponse:
    payload = session_manager.get_status()
    return SessionStatusResponse(**payload)


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket) -> None:
    user = _get_demo_user()
    await websocket.accept()
    try:
        async for frame_payload in session_manager.frame_generator(user.id):
            await websocket.send_json(frame_payload)
    except SessionNotRunningError:
        await websocket.send_json({"running": False})
    except WebSocketDisconnect:  # pragma: no cover - client initiated
        return
    except Exception as exc:  # pragma: no cover - runtime guard
        await websocket.send_json({"running": False, "error": str(exc)})
