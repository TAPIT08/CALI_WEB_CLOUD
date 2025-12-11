from __future__ import annotations

import datetime as dt
from typing import List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserProfile(BaseModel):
    id: str
    email: EmailStr
    display_name: str
    avatar_url: Optional[str]
    member_since: dt.datetime
    level: int
    experience: int
    xp_to_next_level: int


class PreferencesUpdate(BaseModel):
    level: Optional[str] = None
    focus_exercises: Optional[Sequence[str]] = None
    daily_goal: Optional[int] = Field(default=None, ge=1)
    smart_goal: Optional[bool] = None
    timezone: Optional[str] = None
    reminder_hour: Optional[int] = Field(default=None, ge=0, le=23)
    allow_notifications: Optional[bool] = None


class PreferencesResponse(BaseModel):
    level: str
    focus_exercises: List[str]
    daily_goal: int
    smart_goal: bool
    timezone: str
    reminder_hour: int
    allow_notifications: bool


class DashboardResponse(BaseModel):
    user: UserProfile
    preferences: PreferencesResponse
    streaks: dict[str, int]
    progress_timeline: List[dict]
    recommendations: List[dict]
    achievements: List[dict]
    reminders: List[dict]
    suggestions: List[str]


class SessionListResponse(BaseModel):
    sessions: List[dict]


class ReminderCreate(BaseModel):
    message: str = Field(min_length=3)
    remind_at: dt.datetime


class ReminderUpdate(BaseModel):
    completed: bool


class MessageResponse(BaseModel):
    message: str


class StartSessionRequest(BaseModel):
    exercise: str = Field(default="all", pattern="^(pushup|pullup|squat|all)$")
    level: str = Field(default="beginner", pattern="^(beginner|intermediate|advanced)$")
    camera_index: int = Field(default=-1, ge=-1)
    detection_stride: int = Field(default=3, ge=1)
    pose_stride: int = Field(default=1, ge=1)
    show_skeleton: bool = True
    show_metrics: bool = True
    goal_reps: int = Field(default=20, ge=1)


class SessionStartResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    user_id: str
    exercise: str
    goal_reps: int


class SessionStatusResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    running: bool
    message: str
    exercise: Optional[str] = None
    level: Optional[str] = None
    camera: Optional[int] = None
    repCount: Optional[int] = None
    fps: Optional[float] = None
    userId: Optional[str] = None


class SessionStopResponse(BaseModel):
    status: str


class WebsocketFrame(BaseModel):
    timestamp: float
    frame: str
    fps: float
    running: bool
    state: dict
    feedback: List[dict]
    camera: int
    exercise: str
    goalReps: int