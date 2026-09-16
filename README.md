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

The root `main.py` plus the root `backend/` and `frontend/` directories are
the canonical application. The `optivox-web/` tree is retained legacy and
marketing material; it is not part of the current startup path.

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

These batch files are development/exhibition launchers. They bind the backend
to loopback and enable Uvicorn reload. For pilot or production operation, run
the services under a process supervisor with an explicit `OPTIVOX_RUNTIME_MODE`
of `pilot` or `production`, authentication configured, and TLS supplied by a
trusted reverse proxy. Do not use `--reload` for production.

## Runtime security modes

`OPTIVOX_RUNTIME_MODE` must be one of `development`, `exhibition`, `pilot`,
or `production`. Development and exhibition allow the existing loopback
convenience path. Pilot and production do not trust loopback implicitly and
require an API key or configured durable user; production also requires an
administrator authentication method and rejects demo mode.

Validate the current repository’s tracked files for common credential patterns
with:

```bat
python security_scan.py
```

The scanner reports only file names, line numbers, and rule names. It does not
print matched secret values. It is a local guardrail and does not replace a
hosted secret-scanning service.

Run the offline supply-chain checks and generate the two review artifacts with:

```bat
python supply_chain_scan.py
python generate_sbom.py
python generate_model_manifest.py
```

`requirements.txt` and `requirements.lock.txt` pin the reviewed edge baseline
(`torch 2.2.x`/`torchvision 0.17.x` with NumPy 1.26 for the face pipeline),
while `backend/requirements.txt` and
`frontend/package-lock.json` are the reviewed dependency inputs. The model
manifest is generated locally because model weights are intentionally ignored
from Git. Before pilot or production startup, place trusted model files on the
edge device, generate the manifest, and review its hashes. Pilot/production
refuse to start when a configured model is missing, unlisted, or modified;
they never allow Ultralytics to download a replacement silently. This is an
integrity check, not proof that a model is safe, so obtain weights from a
trusted source and record their provenance separately.

InsightFace keeps its `buffalo_l` cache under `~/.insightface` by default. In
strict modes that cache must also contain a local checksum manifest. Generate
it after verifying the trusted files with:

```bat
python generate_model_manifest.py --root "%USERPROFILE%\.insightface\models\buffalo_l" --output model_checksums.json
```

Set `OPTIVOX_INSIGHTFACE_ROOT` if the cache is stored elsewhere. A missing or
unverified face-model cache fails strict startup before `FaceAnalysis` can
invoke its model-availability download path.

Outbound webhooks are HTTPS-only in strict modes and must use the explicit
`OPTIVOX_ALLOWED_WEBHOOK_HOSTS` allowlist. Local HTTP webhooks are supported
only for development/exhibition. Camera sources may be a non-negative local
camera index, a project-contained local video file, or an RTSP/RTSPS URL;
arbitrary URL schemes and path traversal are rejected. Runtime camera health
transitions are appended to `runtime/health_events.jsonl` and shown by the
health API. Run the edge process with a dedicated OS account that can read
models and camera devices, write only its runtime/data directories, and has no
interactive administrator rights. Keep the database, embeddings, alert
credentials, manifests, and backups outside source control and back them up
with restricted filesystem permissions.

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
`OPTIVOX_API_KEY` and, for CLI/edge administration, `OPTIVOX_ADMIN_KEY`.
Browser access uses the authenticated session cookies; do not put an API key
in the Vite build environment. Never commit real values.

## Biometric and evidence storage

Face embeddings are written to a versioned `face_db.secure.json` envelope
instead of the legacy executable-object pickle. The envelope is checksum
verified, written atomically, and uses AES-GCM when `OPTIVOX_BIOMETRIC_KEY` is
set. The key is mandatory in pilot and production modes, but development and
exhibition can use checksum-only compatibility storage. Generate and store
the key outside the repository; losing it makes an encrypted face database unreadable. A legacy `face_db.pkl` is accepted only
through a restricted one-time migration and is removed after the secure file
is written. Raw embeddings are not returned by normal People API responses.

Evidence is restricted to the local snapshots directory, checksummed when
events are recorded, and rejected if a checksum changes. Database backups are
created only through the protected System backup action and include a checksum
manifest. Evidence cleanup uses `OPTIVOX_EVIDENCE_RETENTION_DAYS` and should
be scheduled by the site operator after confirming the retention policy.
Audit records in both SQLite adapters use a chained SHA-256 hash. The System
page exposes the current chain-integrity result; a failure means the audit
history requires investigation before it is trusted.

Automatic attendance follows the configured school calendar. By default,
Monday through Friday are school days. Override this for a pilot site with
`OPTIVOX_SCHOOL_DAYS=0,1,2,3,4` (Monday is `0`, Sunday is `6`) and optionally provide a
comma-separated holiday list with `OPTIVOX_SCHOOL_HOLIDAYS=2026-10-01,2026-12-25`.
Non-school days skip automatic attendance and are not treated as inferred
absence days by the academic overview.
