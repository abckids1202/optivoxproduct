# OptiVox Local Web Application

OptiVox uses one local computer-vision engine, `main.py`, plus a FastAPI backend and React/Vite dashboard. `moretesting.py` is retained as the older standalone runtime and is not the launcher target.

## Architecture

- `main.py` owns the webcam, face recognition, object detection, attendance automation, alerts, snapshots, and SQLite writes.
- `runtime/` is the bridge between the engine and web app.
- `backend/` reads SQLite and runtime files, exposes REST APIs, validates commands, serves snapshots and latest frames, and provides `/ws/live`.
- `frontend/` is the exhibition dashboard.

## URLs

- Frontend: `http://127.0.0.1:5173`
- Backend API: `http://127.0.0.1:8000`
- FastAPI docs: `http://127.0.0.1:8000/docs`

## Startup

Run separately:

```bat
start_optivox.bat
start_backend.bat
start_frontend.bat
```

Or run all three:

```bat
start_all.bat
```

## Runtime Bridge

The engine publishes:

- `runtime/heartbeat.json`: engine/camera/FPS heartbeat.
- `runtime/live_state.json`: current detections, objects, security level, and recent live events.
- `runtime/latest_frame.jpg`: latest annotated frame for the dashboard.
- `runtime/commands.json`: pending web commands written by FastAPI.
- `runtime/command_results.json`: command results written by the engine.
- `runtime/enrollment_status.json`: current enrollment progress.

All JSON writes use a temporary file then atomic replace.

## Demo Mode

The frontend only uses sample data when `VITE_USE_DEMO_DATA=true`. In normal live mode, backend or engine outages are shown explicitly.

## Exhibition Procedure

1. Start `moretesting.py`.
2. Start FastAPI.
3. Start React.
4. Open the frontend in full screen.
5. Confirm the System page shows backend online and engine heartbeat fresh.
6. Use Overview to demonstrate detection, Attendance for records, Security for event history, and People for registration commands.

## Safe Shutdown

Close the frontend tab, stop the backend terminal, then quit the engine with `q` so it can save its shutdown report.

## API protection

Local loopback access works without a key for the exhibition. Before exposing the backend beyond the local machine, set `OPTIVOX_API_KEY` and, for roster administration or attendance corrections, `OPTIVOX_ADMIN_KEY`. Put the matching `VITE_OPTIVOX_API_KEY` in the frontend build environment. Never commit real values.
