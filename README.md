# Real-Time Exercise Form Checker

This project integrates YOLOv8-based exercise identification with MediaPipe Pose-based form analysis to provide real-time feedback for push-ups, pull-ups, and squats. The system supports three skill levels (beginner, intermediate, advanced) with level-appropriate thresholds and coaching cues.

## Features
- Real-time webcam feed ingestion (any OpenCV-compatible camera).
- Interactive configuration panel to pick the target exercise, user level, camera, and UI/audio preferences before every session.
- YOLOv8 exercise detector (supports CPU or GPU execution; CPU by default) running asynchronously to keep the feed responsive.
- MediaPipe Pose tracker with exercise-aware skeleton highlighting that focuses on the primary joints for the current movement.
- Rule-based posture analysis per exercise and skill level with tempo-aware cues.
- Optional pose-skeleton overlay and metric HUD (toggle in the setup panel).
- Rep counting and form validation with instant feedback overlay, smart exercise-switch prompts, and optional audio coaching (voice + beeps).
- Automatic camera discovery plus latency-oriented controls (frame skipping, resolution scaling, optional GPU acceleration).
- Standing posture detection that asks whether to prioritize squat or pull-up coaching.
- Camera-angle guidance when key joints fall out of view.

## Getting Started
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Tip:** Install the matching `torch` build for your GPU if you plan to run inference on CUDA. Otherwise, stick with the CPU wheels (slower but workable with the nano/medium YOLO variants).

### 2. Provide YOLO Weights
Place your trained YOLOv8 exercise detection checkpoint (e.g., `yolov8n-exercise.pt`) under `weights/` as `weights/yolov8n-exercise.pt`.

### 3. Run the Desktop App
```powershell
python -m src.main
```

- Press `q` to exit the preview window.
- Press `1` (prioritize squat) or `2` (prioritize pull-up) when you see the standing prompt.
- When prompted to switch exercises, press `s` to accept the suggestion or ignore to stay on the current focus.
- Add `--device cpu` to force CPU execution or `--device cuda` to hard-pin the GPU.
- Run with `--headless` if you are benchmarking without rendering the overlay.
- Override detection frequency with `--detection-stride 5` (higher stride ⇒ fewer YOLO passes, lower CPU usage).
- Use `--pose-stride 2` to thin pose estimation updates when squeezing extra performance out of the CPU.
- Toggle voice coaching or rep beeps in the setup panel, or disable them entirely via `configs/runtime.yaml` → `audio`.

### 4. Launch the Web Interface
Start the FastAPI service to drive the browser UI and WebSocket stream:

```powershell
uvicorn src.server.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` to access the control panel, start/stop sessions, and view the annotated camera stream directly in the browser.

#### API Overview
- `POST /api/session/start` & `POST /api/session/stop` toggle the background exercise session.
- `GET /api/session/status` reports the current state, camera index, FPS, and rep count.
- `WS /ws/stream` streams annotated JPEG frames plus the latest state/feedback payloads in real time.

The server reuses the existing YOLO, MediaPipe, and rule-based coaching modules. Configuration continues to respect `configs/runtime.yaml` and `configs/exercise_levels.yaml`.

#### Optional Cloud Deployment
Use the provided `Procfile` for platforms that support process types (Heroku, Render, Railway):

1. Commit the project (including `weights/yolov8n-exercise.pt`) to a private repository.
2. Provision the service (e.g., `heroku create --stack container` or a Render Web Service) and connect the repo.
3. Configure build steps to run `pip install -r requirements.txt` and set the start command to `uvicorn src.server.app:app --host 0.0.0.0 --port $PORT` (handled automatically by the Procfile on Heroku-style dynos).
4. Ensure the deployment has access to a camera if live capture is expected; otherwise, run on an edge device or VM with a virtual webcam feed.

For Azure App Service or AWS Elastic Beanstalk:

1. Build a Python 3.11 environment.
2. Upload the repo (ZIP deployment or CI pipeline) with the `weights/` directory intact.
3. Set the startup command to `uvicorn src.server.app:app --host 0.0.0.0 --port 8000`.
4. Configure application settings for camera access or route to a network video stream as needed.

## Configuration Highlights
- Exercise heuristics are defined in `configs/exercise_levels.yaml`.
- Latency and smoothing parameters live in `configs/runtime.yaml`.
- Override the auto camera selection with `--camera 1` (falls back to the first detected device when omitted).
- `configs/runtime.yaml` → `display` block sets the default toggle values for skeleton overlay, metric HUD, smart exercise switching, stat logging, and HUD styling.
- `configs/runtime.yaml` → `audio` controls voice coaching, beeps, and timing/volume preferences.

## Roadmap Ideas
- Session metadata capture (name, age, gender) strictly for logs; exercise & level remain the only factors that alter thresholds or feedback intensity.
- Add curated push-up/pull-up/squat variations while keeping the program scoped to the push–pull–legs trio (or full-body when "all" is selected).
- Incorporate lightweight LSTM to refine rep segmentation.
- Export feedback summaries and rep counts to CSV or a dashboard.

## License
This project is distributed under the MIT License.

### Deploying to Render

This repository includes a Render blueprint (`render.yaml`) for one-click deployment.

- Type: Web Service (Python)
- Build: `pip install -r requirements.txt`
- Start: `uvicorn src.server.app:app --host 0.0.0.0 --port $PORT`
- Health Check: `/`

Steps (Blueprint):
1. Push this repo to GitHub/GitLab/Bitbucket.
2. In Render, click New → Blueprint → paste your repo URL.
3. Review the plan and Create Resources.
4. Render will install dependencies and start the service; open the provided URL.

Steps (Manual Web Service):
1. New → Web Service → Connect repo.
2. Environment: Python.
3. Build Command: `pip install -r requirements.txt`.
4. Start Command: `uvicorn src.server.app:app --host 0.0.0.0 --port $PORT`.
5. Health Check Path: `/`.

Cloud considerations:
- Webcam/audio aren’t available in Render; disable cloud audio in `configs/runtime.yaml` (`audio.enable_tts: false`, `audio.enable_beep: false`).
- GPU isn’t available on standard plans; YOLO/MediaPipe will run on CPU if used server-side.
- Static UI is served under `/static`; the root `/` serves `index.html`.
