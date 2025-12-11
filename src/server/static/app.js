const statusText = document.getElementById("status-text");
const fpsValue = document.getElementById("fps-value");
const repValue = document.getElementById("rep-value");
const feedbackList = document.getElementById("feedback-list");
const streamImage = document.getElementById("stream-image");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const controlForm = document.getElementById("control-form");
const exerciseInput = document.getElementById("exercise");
const levelInput = document.getElementById("level");
const cameraInput = document.getElementById("camera");
const showSkeletonInput = document.getElementById("show-skeleton");
const showMetricsInput = document.getElementById("show-metrics");
const detectionStrideInput = document.getElementById("detection-stride");
const poseStrideInput = document.getElementById("pose-stride");
const dashboardCard = document.getElementById("dashboard-card");
const preferencesCard = document.getElementById("preferences-card");
const controlsCard = document.getElementById("controls-card");
const statusPanel = document.getElementById("status-panel");
const streamCard = document.getElementById("stream-card");
const feedbackPanel = document.getElementById("feedback-panel");
const historyCard = document.getElementById("history-card");
const achievementsCard = document.getElementById("achievements-card");
const remindersCard = document.getElementById("reminders-card");
const dashboardName = document.getElementById("dashboard-name");
const dashboardLevel = document.getElementById("dashboard-level");
const dashboardXp = document.getElementById("dashboard-xp");
const streaksList = document.getElementById("streaks-list");
const suggestionsList = document.getElementById("suggestions-list");
const timelineList = document.getElementById("timeline-list");
const goalValue = document.getElementById("goal-value");
const preferencesForm = document.getElementById("preferences-form");
const prefLevel = document.getElementById("pref-level");
const prefDailyGoal = document.getElementById("pref-daily-goal");
const prefSmartGoal = document.getElementById("pref-smart-goal");
const prefTimezone = document.getElementById("pref-timezone");
const prefReminderHour = document.getElementById("pref-reminder-hour");
const prefAllowNotifications = document.getElementById("pref-allow-notifications");
const focusCheckboxes = Array.from(document.querySelectorAll("#preferences-form .checkbox-grid input[type=\"checkbox\"]"));
const historyList = document.getElementById("history-list");
const achievementsList = document.getElementById("achievements-list");
const remindersList = document.getElementById("reminders-list");
const reminderForm = document.getElementById("reminder-form");
const reminderMessage = document.getElementById("reminder-message");
const reminderDatetime = document.getElementById("reminder-datetime");

let socket = null;
let statusPollTimer = null;
let lastFrameTimestamp = 0;

const protocol = window.location.protocol === "https:" ? "wss" : "ws";

function showSection(section) {
    if (section) {
        section.classList.remove("hidden");
    }
}

function showMainUI() {
    [
        dashboardCard,
        preferencesCard,
        controlsCard,
        statusPanel,
        streamCard,
        feedbackPanel,
        historyCard,
        achievementsCard,
        remindersCard,
    ].forEach(showSection);
}

function formatDateTime(value) {
    if (!value) {
        return "";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return String(value);
    }
    return `${date.toLocaleDateString()} ${date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}

function readNumber(input, fallback) {
    if (!input) {
        return fallback;
    }
    const raw = input.value;
    if (raw === undefined || raw === null || String(raw).trim() === "") {
        return fallback;
    }
    const parsed = Number(raw);
    return Number.isNaN(parsed) ? fallback : parsed;
}

function readText(input, fallback = "") {
    if (!input) {
        return fallback;
    }
    return input.value ?? fallback;
}

function readChecked(input, fallback = false) {
    if (!input) {
        return fallback;
    }
    return Boolean(input.checked);
}

function setRunningState(running) {
    startBtn.disabled = running;
    stopBtn.disabled = !running;
}

function updateStatusPanel(data) {
    if (!data) {
        statusText.textContent = "Idle";
        fpsValue.textContent = "0.0";
        repValue.textContent = "0";
        return;
    }
    statusText.textContent = data.message ?? (data.running ? "Running" : "Idle");
    if (typeof data.fps === "number") {
        fpsValue.textContent = data.fps.toFixed(1);
    }
    if (typeof data.repCount === "number") {
        repValue.textContent = String(data.repCount);
    }
}

function renderFeedback(items) {
    feedbackList.innerHTML = "";
    if (!Array.isArray(items) || items.length === 0) {
        return;
    }
    items.forEach((item) => {
        const container = document.createElement("li");
        container.className = `feedback-item ${item.severity ?? ""}`.trim();
        const message = document.createElement("div");
        message.textContent = item.message;
        container.appendChild(message);
        if (Array.isArray(item.hints) && item.hints.length > 0) {
            const hints = document.createElement("div");
            hints.className = "hints";
            item.hints.forEach((hint) => {
                const hintLine = document.createElement("div");
                hintLine.textContent = hint;
                hints.appendChild(hintLine);
            });
            container.appendChild(hints);
        }
        feedbackList.appendChild(container);
    });
}

async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    const text = await response.text();
    const data = text ? JSON.parse(text) : null;
    if (!response.ok) {
        const detail = data?.detail ?? response.statusText ?? "Request failed";
        throw new Error(detail);
    }
    return data;
}

function populateUserProfile(profile) {
    if (!profile) {
        return;
    }
    dashboardName.textContent = profile.display_name ?? profile.displayName ?? "Athlete";
    dashboardLevel.textContent = `Level ${profile.level ?? 1}`;
    dashboardXp.textContent = `${profile.experience ?? 0} XP  ${(profile.xp_to_next_level ?? 0)} to next level`;
}

function populateStreaks(streaks) {
    streaksList.innerHTML = "";
    if (!streaks) {
        return;
    }
    Object.entries(streaks).forEach(([exercise, value]) => {
        const item = document.createElement("li");
        item.textContent = `${exercise}: ${value} day${value === 1 ? "" : "s"}`;
        streaksList.appendChild(item);
    });
}

function populateSuggestions(suggestions) {
    suggestionsList.innerHTML = "";
    if (!Array.isArray(suggestions) || suggestions.length === 0) {
        const fallback = document.createElement("li");
        fallback.textContent = "Complete a session to unlock personalized tips.";
        suggestionsList.appendChild(fallback);
        return;
    }
    suggestions.forEach((suggestion) => {
        const item = document.createElement("li");
        item.textContent = suggestion;
        suggestionsList.appendChild(item);
    });
}

function populateTimeline(timeline) {
    timelineList.innerHTML = "";
    if (!Array.isArray(timeline) || timeline.length === 0) {
        const empty = document.createElement("li");
        empty.textContent = "No sessions yet. Start one to see your progress.";
        timelineList.appendChild(empty);
        return;
    }
    const seen = new Set();
    timeline.forEach((entry) => {
        if (!entry) {
            return;
        }
        const dateKey = entry.date ?? entry.timestamp ?? entry.ended_at;
        if (!dateKey || seen.has(dateKey)) {
            return;
        }
        seen.add(dateKey);
        const summary = entry.daily_summary ?? entry.dailySummary;
        const totalReps = summary?.total_reps ?? summary?.totalReps ?? entry.reps ?? entry.rep_count ?? 0;
        const readableDate = (() => {
            const parsed = new Date(dateKey);
            if (Number.isNaN(parsed.getTime())) {
                return String(dateKey);
            }
            return parsed.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
        })();

        const item = document.createElement("li");
        const dayHeader = document.createElement("div");
        dayHeader.className = "timeline-day";
        dayHeader.textContent = `${readableDate} • ${totalReps} total reps`;
        item.appendChild(dayHeader);

        const exercisesList = document.createElement("ul");
        exercisesList.className = "timeline-exercises";
        const exercises = summary?.exercises ?? summary?.exerciseTotals;
        const exerciseEntries = exercises && typeof exercises === "object" ? Object.entries(exercises) : null;

        if (exerciseEntries && exerciseEntries.length > 0) {
            exerciseEntries.forEach(([exerciseName, data]) => {
                const reps = data?.reps ?? 0;
                const target = data?.target;
                const accuracy = typeof data?.accuracy === "number" ? Math.round(data.accuracy * 100) : null;
                const consistency = typeof data?.consistency === "number" ? Math.round(data.consistency * 100) : null;
                const parts = [`${exerciseName}: ${reps} reps`];
                if (typeof target === "number") {
                    parts.push(`goal ${target}`);
                }
                if (accuracy !== null) {
                    parts.push(`${accuracy}% acc`);
                }
                if (consistency !== null) {
                    parts.push(`${consistency}% flow`);
                }
                const line = document.createElement("li");
                line.textContent = parts.join(" · ");
                exercisesList.appendChild(line);
            });
        } else {
            const exerciseName = entry.exercise ?? "Session";
            const reps = entry.reps ?? entry.rep_count ?? 0;
            const target = entry.target;
            const accuracy = typeof entry.accuracy === "number" ? Math.round(entry.accuracy * 100) : null;
            const consistency = typeof entry.consistency === "number" ? Math.round(entry.consistency * 100) : null;
            const parts = [`${exerciseName}: ${reps} reps`];
            if (typeof target === "number") {
                parts.push(`goal ${target}`);
            }
            if (accuracy !== null) {
                parts.push(`${accuracy}% acc`);
            }
            if (consistency !== null) {
                parts.push(`${consistency}% flow`);
            }
            const line = document.createElement("li");
            line.textContent = parts.join(" · ");
            exercisesList.appendChild(line);
        }

        item.appendChild(exercisesList);
        timelineList.appendChild(item);
    });
}

function updateGoalPreview() {
    if (!goalValue) {
        return;
    }
    const value = readNumber(prefDailyGoal, 0);
    goalValue.textContent = `${value} reps`;
}

function populatePreferences(preferences) {
    if (!preferences) {
        return;
    }
    if (prefLevel) {
        prefLevel.value = preferences.level;
    }
    if (prefDailyGoal) {
        prefDailyGoal.value = preferences.daily_goal;
    }
    if (prefSmartGoal) {
        prefSmartGoal.checked = Boolean(preferences.smart_goal);
    }
    if (prefTimezone) {
        prefTimezone.value = preferences.timezone;
    }
    if (prefReminderHour) {
        prefReminderHour.value = preferences.reminder_hour;
    }
    if (prefAllowNotifications) {
        prefAllowNotifications.checked = Boolean(preferences.allow_notifications);
    }
    const focus = new Set(preferences.focus_exercises ?? []);
    focusCheckboxes.forEach((checkbox) => {
        checkbox.checked = focus.size === 0 ? checkbox.checked : focus.has(checkbox.value);
    });
    updateGoalPreview();
}

function populateHistory(sessions) {
    historyList.innerHTML = "";
    if (!Array.isArray(sessions) || sessions.length === 0) {
        const empty = document.createElement("li");
        empty.textContent = "No recorded sessions yet.";
        historyList.appendChild(empty);
        return;
    }
    sessions.forEach((session) => {
        const item = document.createElement("li");
        item.textContent = `${formatDateTime(session.ended_at || session.started_at)}  ${session.exercise}  ${session.rep_count} reps  ${Math.round((session.accuracy ?? 0) * 100)}% accuracy`;
        historyList.appendChild(item);
    });
}

function populateAchievements(achievements) {
    achievementsList.innerHTML = "";
    if (!Array.isArray(achievements) || achievements.length === 0) {
        const empty = document.createElement("li");
        empty.textContent = "No achievements yet. Stay consistent to earn badges.";
        achievementsList.appendChild(empty);
        return;
    }
    achievements.forEach((achievement) => {
        const item = document.createElement("li");
        item.innerHTML = `<strong>${achievement.name}</strong>  ${achievement.description}  ${formatDateTime(achievement.earned_at)}`;
        achievementsList.appendChild(item);
    });
}

async function completeReminder(reminderId) {
    try {
        await fetchJson(`/api/reminders/${reminderId}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ completed: true }),
        });
        await loadReminders();
    } catch (err) {
        console.error("Failed to complete reminder", err);
    }
}

function renderReminders(reminders) {
    remindersList.innerHTML = "";
    if (!Array.isArray(reminders) || reminders.length === 0) {
        const empty = document.createElement("li");
        empty.textContent = "No reminders scheduled.";
        remindersList.appendChild(empty);
        return;
    }
    reminders.forEach((reminder) => {
        const item = document.createElement("li");
        item.dataset.id = reminder.id;
        const message = document.createElement("div");
        message.textContent = reminder.message;
        const schedule = document.createElement("div");
        schedule.textContent = `Reminds at ${formatDateTime(reminder.remind_at)}`;
        item.appendChild(message);
        item.appendChild(schedule);
        if (!reminder.completed) {
            const completeBtn = document.createElement("button");
            completeBtn.type = "button";
            completeBtn.className = "secondary";
            completeBtn.textContent = "Mark done";
            completeBtn.addEventListener("click", () => completeReminder(reminder.id));
            item.appendChild(completeBtn);
        } else {
            const badge = document.createElement("span");
            badge.textContent = "Completed";
            item.appendChild(badge);
        }
        remindersList.appendChild(item);
    });
}

async function loadDashboard() {
    const data = await fetchJson("/api/dashboard");
    populateUserProfile(data.user);
    populatePreferences(data.preferences);
    populateStreaks(data.streaks);
    populateSuggestions(data.suggestions);
    populateTimeline(data.progress_timeline ?? data.progressTimeline);
    populateAchievements(data.achievements);
    renderReminders(data.reminders);
    showMainUI();
}

async function loadSessions() {
    const data = await fetchJson("/api/sessions/history");
    populateHistory(data.sessions ?? []);
}

async function loadReminders() {
    const data = await fetchJson("/api/reminders");
    renderReminders(data);
}

function closeSocket() {
    if (socket) {
        socket.close();
        socket = null;
    }
}

function openSocket() {
    closeSocket();
    const wsUrl = new URL("/ws/stream", window.location.origin);
    wsUrl.protocol = protocol;
    socket = new WebSocket(wsUrl.toString());
    socket.onopen = () => {
        console.debug("WebSocket connected");
    };
    socket.onclose = () => {
        console.debug("WebSocket disconnected");
        socket = null;
        setRunningState(false);
        updateStatusPanel({ message: "Idle", running: false });
    };
    socket.onerror = (event) => {
        console.error("WebSocket error", event);
    };
    socket.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.frame && payload.timestamp !== lastFrameTimestamp) {
            lastFrameTimestamp = payload.timestamp;
            streamImage.src = `data:image/jpeg;base64,${payload.frame}`;
        }
        if (typeof payload.fps === "number") {
            fpsValue.textContent = payload.fps.toFixed(1);
        }
        if (payload.state && typeof payload.state.rep_count === "number") {
            repValue.textContent = String(payload.state.rep_count);
        }
        if (Array.isArray(payload.feedback)) {
            renderFeedback(payload.feedback);
        }
        if (payload.running === false && !payload.frame) {
            setRunningState(false);
            statusText.textContent = payload.error ? `Error: ${payload.error}` : "Idle";
        } else {
            statusText.textContent = payload.exercise ? `Running (${payload.exercise})` : "Running";
        }
    };
}

async function refreshStatus() {
    try {
        const data = await fetchJson("/api/session/status");
        updateStatusPanel(data);
        setRunningState(Boolean(data.running));
        if (data.running && !socket) {
            openSocket();
        }
    } catch (err) {
        console.error("Status refresh failed", err);
    }
}

function startStatusPolling() {
    stopStatusPolling();
    statusPollTimer = window.setInterval(refreshStatus, 5000);
    refreshStatus();
}

function stopStatusPolling() {
    if (statusPollTimer) {
        window.clearInterval(statusPollTimer);
        statusPollTimer = null;
    }
}

async function startSession(event) {
    event.preventDefault();
    const payload = {
        exercise: readText(exerciseInput, "all"),
        level: readText(levelInput, "beginner"),
        camera_index: readNumber(cameraInput, -1),
        show_skeleton: readChecked(showSkeletonInput, true),
        show_metrics: readChecked(showMetricsInput, true),
        detection_stride: readNumber(detectionStrideInput, 3),
        pose_stride: readNumber(poseStrideInput, 1),
        goal_reps: readNumber(prefDailyGoal, 20),
    };
    try {
        await fetchJson("/api/session/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        setRunningState(true);
        statusText.textContent = "Starting...";
        openSocket();
    } catch (err) {
        console.error(err);
        alert(`Could not start session: ${err.message}`);
    }
}

async function stopSession() {
    try {
        await fetchJson("/api/session/stop", { method: "POST" });
    } catch (err) {
        console.error(err);
    } finally {
        closeSocket();
        setRunningState(false);
        statusText.textContent = "Idle";
        streamImage.src = "";
        renderFeedback([]);
    }
}

async function handlePreferencesSubmit(event) {
    event.preventDefault();
    const focusExercises = focusCheckboxes.filter((checkbox) => checkbox.checked).map((checkbox) => checkbox.value);
    const payload = {
        level: readText(prefLevel, "beginner"),
        focus_exercises: focusExercises,
        daily_goal: readNumber(prefDailyGoal, 20),
        smart_goal: readChecked(prefSmartGoal, true),
        timezone: readText(prefTimezone, "UTC"),
        reminder_hour: readNumber(prefReminderHour, 9),
        allow_notifications: readChecked(prefAllowNotifications, true),
    };
    try {
        const updated = await fetchJson("/api/users/preferences", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        populatePreferences(updated);
        alert("Preferences saved!");
    } catch (err) {
        console.error("Failed to save preferences", err);
        alert(`Could not save preferences: ${err.message}`);
    }
}

async function handleReminderSubmit(event) {
    event.preventDefault();
    const message = reminderMessage.value.trim();
    const when = reminderDatetime.value;
    if (!message || !when) {
        return;
    }
    try {
        await fetchJson("/api/reminders", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, remind_at: new Date(when).toISOString() }),
        });
        reminderForm.reset();
        await loadReminders();
    } catch (err) {
        console.error("Failed to schedule reminder", err);
        alert(`Could not schedule reminder: ${err.message}`);
    }
}

function initializeEventListeners() {
    controlForm?.addEventListener("submit", startSession);
    stopBtn?.addEventListener("click", stopSession);
    preferencesForm?.addEventListener("submit", handlePreferencesSubmit);
    reminderForm?.addEventListener("submit", handleReminderSubmit);
    prefDailyGoal?.addEventListener("input", updateGoalPreview);
    window.addEventListener("beforeunload", () => {
        closeSocket();
        stopStatusPolling();
    });
}

async function initializeApp() {
    initializeEventListeners();
    try {
        await Promise.all([loadDashboard(), loadSessions()]);
    } catch (err) {
        console.error("Initial load failed", err);
        alert(`Failed to load dashboard data: ${err.message}`);
    }
    startStatusPolling();
}

initializeApp();
