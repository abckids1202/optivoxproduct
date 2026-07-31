# OptiVox Web — Architecture Design Document

This document explains how the local OptiVox cv2 system becomes a web-connected platform,
how streaming and the data path work, and how to evolve the monolithic single-file system
into a maintainable service.

---

## 1. System overview

OptiVox has two tiers that share one SQLite database:

1. **Local AI tier** — your single-file `VisionSystem` and its components
   (`FaceAnalyzer`, `ObjectDetector`, `DangerDetector`, `PoseDetector`, `AntiSpoofDetector`,
   `BehaviorAnalyzer`, `CrowdIntelligence`, `AttendanceManager`, `AIAssistant`).
2. **Web tier** — Flask app, the vision **bridge**, the frame **buffer**, and a read-only DB
   accessor.

```
┌──────────────────────────── HOST MACHINE ────────────────────────────┐
│                                                                       │
│   Camera / RTSP                                                       │
│        │                                                              │
│        ▼                                                              │
│   ┌─────────────────────────── core/vision_bridge.py ─────────────┐  │
│   │  VisionBridge worker thread (SINGLE instance)                  │  │
│   │    loop: read → VisionSystem.process() → log events            │  │
│   │          → attendance.handle_recognition() → JPEG encode       │  │
│   └───────────────┬───────────────────────────────┬───────────────┘  │
│                   │ publishes JPEG                 │ writes            │
│                   ▼                                ▼                   │
│         ┌──── core/stream.py ────┐         ┌──── security.db ────┐    │
│         │  FrameBuffer (1 frame) │         │  events, people,    │    │
│         └──────────┬─────────────┘         │  attendance, ...    │    │
│                    │ reads                 └──────────┬──────────┘    │
│                    ▼                                  │ read-only     │
│         ┌──── blueprints/api.py ────────────────┐     ▼               │
│         │  /api/video_feed  (MJPEG generator)   │  core/db.py         │
│         │  /api/stats /events /attendance       │                     │
│         └──────────┬────────────────────────────┘                     │
│                    │ HTTP                                              │
└────────────────────┼──────────────────────────────────────────────────┘
                     ▼
              Browser dashboard (MJPEG <img> + polled JSON)
```

---

## 2. The streaming pipeline (how OpenCV frames reach the browser)

There are four common ways to get OpenCV frames into a web page. OptiVox uses **MJPEG** by
default because it is the simplest robust option and works in every browser with a plain
`<img>` tag.

| Method | How it works | Latency | Pros | Cons |
|--------|--------------|---------|------|------|
| **MJPEG** (default) | `multipart/x-mixed-replace`; server pushes JPEG frames | Low–med | Trivial client (`<img>`), universal | Higher bandwidth (no inter-frame compression) |
| **WebSocket + base64** | Push frames over a socket, draw to `<canvas>` | Low | Bidirectional, overlay control client-side | More client code, base64 overhead |
| **WebRTC** | Peer media stream | Lowest | True video codec, lowest bandwidth | Complex signalling (STUN/TURN), heavier setup |
| **RTSP** | IP-camera native protocol | n/a in browser | Source format | Browsers can't play RTSP directly; must transcode |

### MJPEG mechanics (what the code does)
- The worker calls `cv2.imencode(".jpg", display, [JPEG_QUALITY])` once per processed frame.
- The encoded bytes are stored in `FrameBuffer` (`core/stream.py`) — only the **latest** frame
  is kept; old frames are discarded.
- `mjpeg_generator()` yields each new frame wrapped in a multipart boundary. Multiple viewers
  each get their own generator but all read the same buffer, so **inference is not duplicated**.
- The dashboard simply uses `<img src="/api/video_feed">`.

### Bandwidth & latency considerations
- Bandwidth ≈ `frame_size_kb × STREAM_FPS × viewers`. Tune `OPTIVOX_JPEG_QUALITY` (default 70)
  and `OPTIVOX_STREAM_FPS` (default 20) down for remote/mobile viewers.
- End-to-end latency is dominated by `VisionSystem.process()` time, not transport. The bridge
  records `process_ms` in the frame metadata (`/api/health`).
- For many simultaneous remote viewers or sub-100ms latency, migrate the stream to **WebRTC**
  (see Roadmap) and keep the buffer pattern unchanged.

### Security implications of streaming
- `/api/video_feed` is behind `@login_required`. Never expose it unauthenticated.
- Put TLS in front (reverse proxy) so the feed isn't sent in clear text.
- Treat the stream as sensitive PII (it shows faces). Restrict by org and log access.

---

## 3. The data path (events & attendance)

The bridge mirrors the essential per-frame logic from your `main()` loop, minus the desktop
`cv2.imshow`/keyboard UI:

```python
(display, faces_info, hand_dets, obj_dets, events, tracked_count, dt) = vision.process(frame)

# attendance: every recognized real, non-stranger face
for fi in faces_info:
    if fi["is_real"] and fi["name"] not in ("UNKNOWN","SPOOF") and not fi["name"].startswith("STRANGER_"):
        attendance.handle_recognition(fi["name"], cam_id, cam_location)

# events: log each with mapped severity (snapshots/alerts can be added here)
for etype, target, conf, details in events:
    db.log_event(etype, person_id=resolve(target), confidence=conf,
                 details=details, camera_id=cam_id, location=cam_location,
                 severity=_SEVERITY_MAP.get(etype, 0))
```

The web API then reads those rows **read-only** (`core/db.py` opens the DB with `mode=ro`), so
the dashboard can never corrupt the operational database.

> The bridge intentionally does **not** duplicate alerts/snapshots/voice. Add those calls inside
> `VisionBridge._handle_events` if you want the web worker to drive them too — your
> `AlertManager` and `_save_snapshot` are import-compatible.

---

## 4. Authentication & organization access

- Session-based login (`blueprints/auth.py`) with a `@login_required` decorator guarding the
  dashboard and all `/api/*` data + stream routes (except `/api/contact` and `/api/health`).
- The current build ships a **single demo org** (`OPTIVOX_USER`/`OPTIVOX_PASS`).
- **Production hardening** (do before real use):
  - Replace with a user table; store `werkzeug.security.generate_password_hash` hashes.
  - Add per-org scoping (org_id on every query) for multi-tenant SaaS.
  - Optional JWT for API clients; CSRF protection on form posts (Flask-WTF);
    rate limiting (Flask-Limiter); secure session cookies (`SESSION_COOKIE_SECURE`).

---

## 5. API surface

| Route | Method | Auth | Purpose |
|-------|--------|------|---------|
| `/` `/features` `/demo` `/about` `/faq` `/contact` | GET | public | Marketing pages |
| `/login` | GET/POST | public | Org login |
| `/logout` | GET | — | Clear session |
| `/dashboard/` | GET | required | Command center |
| `/api/video_feed` | GET | required | MJPEG live stream |
| `/api/stats` | GET | required | KPI counts (faces, events today, clocked-in, strangers) |
| `/api/events?limit=N` | GET | required | Recent events with severity |
| `/api/attendance` | GET | required | Today's attendance records |
| `/api/people` | GET | required | Enrolled people |
| `/api/health` | GET | public | Bridge mode + stream metadata |
| `/api/contact` | POST | public | Request-access form intake |

---

## 6. Database schema (owned by the local system)

The web layer **reads** the schema your single-file system creates. Key tables:

- `people(id, name, role, thumbnail_path, metadata_json, created_at, updated_at)`
- `events(id, person_id, event_type, confidence, details_json, snapshot_path, camera_id, location, severity, timestamp)`
- `attendance(id, person_id, date, clock_in, clock_out, work_minutes, late_minutes, camera_id, location, notes)`
- `event_stats_daily`, `event_stats_hourly`, `behavior_profiles`, `zone_activity`, `alert_log`, `audit_log`

Face **embeddings** live in `data/face_db.pkl` (pickle), not in SQLite — the web layer never
needs them. Migrating embeddings to a vector column (PostgreSQL + pgvector) is a future option.

---

## 7. Refactoring the monolith (recommended)

Your single file is ~2,500 lines doing everything. It works, but to make it web-serviceable
and testable, split it along the seams that already exist as classes:

```
optivox/
├── config.py             # the CONFIG dict + class maps
├── db.py                 # EventDatabase, DatabaseMigrationManager
├── vision/
│   ├── faces.py          # FaceAnalyzer, FAISSIndexer
│   ├── objects.py        # ObjectDetector, DangerDetector, SupplementaryDetector, CustomObjectManager
│   ├── pose_hands.py     # PoseDetector, HandDetector
│   ├── antispoof.py      # AntiSpoofDetector + helpers
│   ├── behavior.py       # BehaviorAnalyzer, SuspicionScorer, CrowdIntelligence, CentroidTracker
│   └── system.py         # VisionSystem (orchestrator)
├── attendance.py         # AttendanceManager
├── alerts.py             # AlertManager, VoiceManager
├── assistant.py          # AIAssistant + TOOL_DEFINITIONS
└── desktop_app.py        # the old main() loop (optional, for the cv2 window)
```

The web bridge imports `vision.system.VisionSystem`, `db.EventDatabase`, and
`attendance.AttendanceManager` — exactly the seam this platform already targets. Until you
refactor, the bridge works against the single file as long as it exposes those names at module
top level (which it does).

### Async / threading / performance
- Keep capture + inference on **one** worker thread; CPython's GIL means extra threads won't
  speed up the numpy/onnx work, and concurrent `process()` calls would thrash the models.
- For higher throughput: run heavy detectors on a **GPU** (swap `CPUExecutionProvider` →
  `CUDAExecutionProvider` for InsightFace; YOLO auto-uses CUDA), and keep the existing
  `EVERY_N_FRAMES` strides in `CONFIG["PERFORMANCE"]`.
- If you later need multiple cameras, run one bridge worker **per camera**, each with its own
  `FrameBuffer`, and add a `camera_id` path parameter to `/api/video_feed`.

---

## 8. Scalability outlook

| Stage | Change |
|-------|--------|
| Single camera (now) | One bridge worker + MJPEG |
| Multi-camera | One worker + buffer per camera; camera selector in dashboard |
| Many remote viewers | Switch transport to WebRTC; keep buffer pattern |
| Multi-org SaaS | Per-org auth + DB scoping; PostgreSQL; Redis for live state |
| Cloud inference | Containerize detectors as microservices; gRPC frame queue |
| Edge devices | Run bridge on Jetson / Pi; stream to a central dashboard |
```
