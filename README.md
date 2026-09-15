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
- `runtime/capability.json`: runtime version, process identity, and available engine capabilities.
- `runtime/commands.json`: pending web commands written by FastAPI.
- `runtime/command_results.json`: command results written by the engine.
- `runtime/enrollment_status.json`: current enrollment progress.

All JSON writes use a temporary file then atomic replace.

## Performance Runtime

The live engine uses four bounded responsibilities:

- capture owns `cv2.VideoCapture` and publishes one latest frame with a monotonic frame ID;
- inference consumes the newest available frame and records stage timing, frame age, and stale-frame drops;
- the display loop renders the newest completed result at a bounded UI rate;
- operations persist attendance/events and dispatch alerts from bounded regular and critical queues.

Performance metrics are included in `runtime/heartbeat.json` and `runtime/live_state.json` under `performance`. A compact final snapshot is written to `runtime/performance_summary.json` when the engine exits. It includes capture/display/inference rates, frame IDs, frame-age percentiles, accurate replacement/skip counters, optional process resource telemetry, and per-subsystem call/latency metrics. Use the reported values before choosing any later model optimization.

Sprint II adds adaptive active/idle schedules, stable-track recognition reuse, live face-quality gating, and padded person ROIs for hand/pose inference. Gate III makes confirmed-track reuse bypass face embedding until revalidation, records explicit identity state and fresh attendance evidence, adds per-model workload counters, and separates `CAPTURE_*`, `PROCESSING_*`, and `DISPLAY_*` resolution settings. When processing is downscaled, global inference uses the processing frame while face quality, liveness, and embeddings use mapped coordinates on the original capture frame. The general object detector and face detector remain full-frame. `ENABLE_*_INFERENCE` switches control compute; `SHOW_*` switches only control overlays. GPU/VRAM telemetry remains `NOT MEASURED` unless a hardware-specific collector is added.

## Demo Mode

The frontend only uses sample data when `VITE_USE_DEMO_DATA=true`. In normal live mode, backend or engine outages are shown explicitly.

## Exhibition Procedure

1. Start `main.py` (the canonical vision runtime).
2. Start FastAPI.
3. Start React.
4. Open the frontend in full screen.
5. Confirm the System page shows backend online and engine heartbeat fresh.
6. Use Overview to demonstrate detection, Attendance for records, Security for event history, and People for registration commands.

## Safe Shutdown

Close the frontend tab, stop the backend terminal, then quit the engine with `q` so it can save its shutdown report.

## Benchmark

With the engine running, collect a non-invasive ten-minute runtime report:

```bat
python benchmark_runtime.py --seconds 600 --publish-runtime-summary
```

The observer writes a timestamped JSON report under `reports/` and, with
`--publish-runtime-summary`, updates `runtime/performance_summary.json` for
the dashboard performance endpoint. It samples
capture, inference, display, frame age, latency, CPU, queue, cache, and
recognition counters without opening the camera a second time or writing to
the operational database. GPU and VRAM remain `NOT_MEASURED` unless a
hardware-specific collector is configured.

## API protection

Local loopback access works without a key for the exhibition. Sensitive data
and live-stream routes require operator authentication when keys are
configured. Before exposing the backend beyond the local machine, set
`OPTIVOX_API_KEY` and, for roster administration or attendance corrections,
`OPTIVOX_ADMIN_KEY`. Put the matching `VITE_OPTIVOX_API_KEY` in the frontend
build environment. Never commit real values.
