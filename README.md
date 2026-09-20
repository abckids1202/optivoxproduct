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

Strict pilot and production startup also require
`CORRELATION_CORE.ENABLED=true`. This prevents a deployment from silently
falling back to raw frame labels for attendance or security decisions;
development and exhibition may still use an explicit compatibility fallback
for troubleshooting.

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

`runtime/live_state.json` includes a sanitized `attendance.center_mode` view of
the automatic center gate: phase, candidate, progress, liveness phase/status,
and challenge attempts. It is operational state only; the underlying edge
process keeps embeddings and raw biometric material local.

It also exposes labelled crowd telemetry from tracked-person occupancy,
including grid cells, short-term growth, and movement speed. Perspective
correction and physical density remain `NOT_CONFIGURED`/`NOT_MEASURED` until a
camera calibration and evaluation set exist.

Liveness evaluation metrics are published separately in
`runtime/liveness_metrics.json`. The report includes challenge attempts,
completion time, timeouts, aggregate false-accept/false-reject rates, and
per-scenario rates. A scenario label such as `printed_photo`, `phone_screen`,
`replay_video`, `lighting_changes`, `glasses`, `mask`, `occlusion`, or
`multi_person` is evidence metadata only; it becomes a measured result only
after a consented, labelled evaluation is recorded. The current runtime does
not invent those labels or claim that heuristic liveness is security-grade.

Guided enrollment is controlled from the People page. The edge runtime collects
quality-gated samples while the page polls `enrollment_status.json`; after the
minimum sample count is reached, `Finish capture` queues `finish_enrollment`
without closing the camera. The operator then reviews the quality and duplicate
warning before `confirm_enrollment` commits the identity. Rejected samples,
quality scores, pose coverage, and provenance remain part of the local report.

The backend operations API exposes the entity-scoped liveness audit trail at
`/api/operations/liveness-challenges`. It reports challenge phase, outcome,
timing, attempt number, entity/session, and source-frame provenance. It uses
the operations permission and filters biometric material; embeddings, raw
face crops, and templates are never returned in this response.

Security signals follow the same boundary. The first correlation pass reduces
behavior, crowd, pose, and zone outputs after identity state is available; a
second final pass enriches late object, spoof, and interaction signals before
the operations worker sees them. Person events may link to an entity and
presence session, while object/system events explicitly remain global. This
prevents a global alert from being silently attributed to the only visible
person.

The pre-persistence envelope deliberately does not invent a database
`presence_session_id`. It carries the stable entity and generation context;
the operations worker attaches the actual presence-session row ID after its
idempotent upsert. This keeps correlation context and persisted foreign keys
distinct.

The correlation reducer is also connected to the active SQLite roster. A
confirmed model label is not automatically an authorized identity: the
canonical entity snapshot includes `roster_match` and `roster_validation`,
and the final attendance gate requires an active roster match. Disabled or
deleted people may remain visible as continuity state briefly, but their
attendance is rejected and their named security attribution is downgraded to
`UNKNOWN` until the roster is refreshed.

Correlated security events carry a stable `event_uid` into the local database.
The events table keeps that value unique when present, so an operations retry
is idempotent and cannot duplicate the event or its daily/hourly counters.
Stable cached recognition may continue refreshing an already-open attendance
presence record, but only while the correlated entity remains visibly eligible
with fresh identity evidence and `REAL` liveness.

Security incidents are now materialized by the edge operations worker after a
correlated event is persisted and its alert attempt has completed. The runtime
uses the same context keys as the backend incident service (category, entity,
camera, zone, presence session, correlation ID, and a ten-minute window), links
the source event, evidence, and alert delivery, and remains idempotent when an
operations task is replayed. The dashboard can therefore review incidents while
the backend is offline; the backend service remains responsible for filtering,
normalizing, assignment, and human review actions.

Vehicle speed is treated as a calibrated measurement, not a pixel-motion claim.
The runtime maps detector coordinates from the active processing resolution into
the calibration image resolution before projecting onto the ground plane. It
emits bounded `SPEED_ESTIMATE` events only when calibration is valid; otherwise
the capability remains `NOT_CONFIGURED`. Plate handling remains a restricted
temporal-voting adapter and reports uncertainty until a licensed plate/OCR
adapter is explicitly installed.

## Performance Runtime

The live engine uses four bounded responsibilities:

- capture owns `cv2.VideoCapture` and publishes one latest frame with a monotonic frame ID;
- inference consumes the newest available frame and records stage timing, frame age, and stale-frame drops;
- the display loop renders the newest completed result at a bounded UI rate;
- operations persist attendance/events and dispatch alerts from bounded regular and critical queues.

Performance metrics are included in `runtime/heartbeat.json` and `runtime/live_state.json` under `performance`. A compact final snapshot is written to `runtime/performance_summary.json` when the engine exits. It includes capture/display/inference rates, frame IDs, frame-age percentiles, accurate replacement/skip counters, optional process resource telemetry, and per-subsystem call/latency metrics. Use the reported values before choosing any later model optimization.

Frame age is reported at three decision points: the newest frame consumed by
inference, the newest frame that completed inference, and frames rejected as
stale. The backend exposes consumed and stale-frame p95 values separately so a
low average cannot hide a tail of unusably old frames.

`benchmark_runtime.py` refuses to sample a heartbeat older than ten seconds or
one whose engine/camera is not active, so an offline or previous run cannot be
reported as a fresh benchmark. A sample is counted as fresh-frame evidence
only when the source frame ID is numeric and strictly increasing; a changing
heartbeat without frame progress is `NOT_MEASURED`. A collection with fewer
than two fresh samples is marked `PARTIAL`. The default ten-minute duration is
also required before the report can be `MEASURED`/`COMPLETE`; use
`--required-seconds` only when a different validation protocol is explicitly
approved. Missing stage or hardware telemetry remains `NOT_MEASURED`. The
report also records whether checks were active, stale, inactive, or invalid,
including large future clock skew, plus cumulative frame,
queue, recognition, and cache counters. A cache-hit percentage is a runtime
efficiency signal, not a recognition-accuracy score.

The deterministic replay harness treats `active_roster` as an authorization
input. When present, it is configured on `CorrelationCore` before replayed
security or attendance decisions; roster membership is therefore measured as
part of the decision path rather than applied only after an unrestricted gate.

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

Model governance is separate from the checksum manifest. An optional
`models/model_registry.json` records each artifact's task, version, source,
license, evaluation report, promotion state, and rollback target. The runtime
validates it once at startup and exposes `model_registry` in capabilities; it
never downloads or replaces a model automatically. Missing or invalid registry
metadata is reported as `NOT_CONFIGURED` or `INVALID`, not as a promoted model.

The canonical runtime also exposes a model-adapter snapshot in
`runtime/live_state.json` and performance telemetry. The adapter boundary
describes task, version, cadence, ROI policy, resource budget, privacy class,
execution mode, failures, and latency for the existing face, YOLO, danger,
pose, hand, tracking, liveness, crowd, vehicle, plate, demographic, and custom
object components. `plate_detector_ocr` remains `NOT_CONFIGURED` until a
licensed local detector/OCR pair is installed and evaluated. Adapter states
such as `FAILED_INTEGRITY` and `FAILED_RUNTIME` are capability signals; they
cannot create attendance or resolve incidents directly. The person tracker is
behind a replaceable tracking boundary, while `CorrelationCore` remains the
authority for identity, attendance, and security decisions. A model marked
`shadow` or `canary` is observable but is not approved for operational
selection; promotion additionally requires a verified checksum and evaluation
report. `rollback` is fail-closed and is visible in the System dashboard.
At startup, matching adapter names are synchronized with the local model
registry; unmatched built-in components remain explicitly `not_applicable`.
The side-effect-free `core/shadow_runner.py` can compare a baseline adapter
with a candidate on supplied frames, report disagreement and latency, and
return `decision_effect: NONE`; it never forwards candidate output to
attendance, incidents, alerts, permissions, or biometric operations.
The expected shape is:

```json
{
  "schema_version": 1,
  "models": [{
    "name": "face-recognizer",
    "task": "face_embedding",
    "version": "1.0.0",
    "path": "models/face.onnx",
    "sha256": "<64-character-sha256>",
    "source": "documented-source",
    "license": "documented-license",
    "evaluation_report": "reports/face-evaluation.json",
    "promotion_status": "shadow",
    "rollback_target": ""
  }]
}
```

## Evaluation and Vehicle Intelligence

The repository includes a provenance-aware dataset registry and dependency-light
metrics for the next evaluation cycle. A manifest is JSON with `schema_version`
`1`, a dataset `root`, and records containing `record_id`, relative `path`,
`task`, `split`, and `provenance`. Supported splits are `train`, `validation`,
`holdout`, and `adversarial`; source IDs must not cross splits. Generated
records also include a streaming `content_sha256` checksum, which is verified
when files are required. Validate one with:

```bat
python dataset_registry.py data/dataset_manifest.json --require-files
```

The registry rejects paths outside the dataset root, missing consent/licence
provenance, derived samples without transformation metadata, exact duplicate
media, broken derived-record lineage, derived samples moved into a different
split, source leakage between evaluation splits, and post-registration media
tampering through the recorded checksum. It stores metadata only;
it does not copy or upload images. Runtime augmentation is bounded and
provenance-tagged: derived embeddings can support matching, but only original
enrollment evidence calibrates the per-person threshold.

Vehicle tracking is integrated into the canonical runtime through the existing
object detector. It reports vehicle tracks plus debounced `VEHICLE_ENTERED` and
timeout-based `VEHICLE_EXITED` observations.
Physical speed is `NOT_CONFIGURED` until a valid image-to-ground homography is
provided under `VEHICLE_INTELLIGENCE.CALIBRATION`; pixels-per-second is never
presented as physical speed. A future licensed plate detector/OCR adapter may
submit OCR results to the temporal voting boundary in `core/vehicle_adapters.py`, which emits only a stable
`PLATE_READ` with a restricted-local hash and confidence. Unreadable crops can
be represented as `PLATE_READ_UNCERTAIN`; the runtime does not invent
characters. Raw plate text is removed again at the operational persistence
boundary.

An authenticated system operator can install a validated calibration profile by
POSTing a 3x3 homography and source image dimensions to
`/api/system/vehicle-calibration`. The command is queued to the local runtime,
stored under `runtime/vehicle_calibration.json`, and audited. Invalid or
non-finite profiles are rejected and leave physical speed unavailable.

The current vehicle capability intentionally reports plate reading as
`NOT_CONFIGURED`. No plate model or external dataset is downloaded silently.
Running detection can use a frame-normalized velocity threshold when the
runtime supplies processing dimensions; otherwise it explicitly falls back to
the legacy pixel threshold and records `threshold_mode=pixel_fallback` in the
event metadata. Evacuation detection follows the same contract and records
the crowd speed, threshold, frame diagonal, and threshold mode. Neither mode
is a physical speed measurement; calibrated ground-plane mapping is still
required for vehicle speed.
Use the evaluation metrics in `core/evaluation.py` to record false accepts,
false rejects, event false-alert rates, tracking ID switches, and calibrated
vehicle speed error before promoting new models or policies.

The deterministic policy replay harness exercises correlation, attendance
gates, presence transitions, and security-zone decisions without opening a
camera:

```bat
python replay_validation.py data/replay_example.json --output reports/replay.json
```

Replay is a policy/integration check, not a claim about model accuracy. Use
holdout and adversarial recordings for recognition, liveness, detection, and
vehicle accuracy metrics.

### Versioned decision policy

Operational decisions are stamped with a content-addressed policy identity.
`core/policy_engine.py` builds the snapshot from attendance, security,
correlation, notification, and privacy configuration, validates the school
calendar and numeric thresholds, and exposes the result in
`runtime/capability.json` under `capabilities.policy`. Recognition evidence,
attendance decisions, correlation results, security observations, and replay
reports carry the same `policy_version` so an operator can reconstruct which
rules produced a decision. A snapshot with validation issues is visible as
invalid and must be corrected before it is treated as a trusted site policy.
The shared SQLite database retains each scoped snapshot in `policy_snapshots`
with immutable content and a last-seen timestamp. Authorized operators can
inspect the redacted history through `GET /api/system/policy`; credentials and
private webhook configuration are never returned by that endpoint.

For a repeatable validation gate, place multiple scenario JSON files in a
directory and run:

```bat
python evaluate_replay_suite.py data/replay-suite --output reports/evaluation-suite.json
```

Promotion thresholds are optional and must be chosen for the site risk policy;
the repository includes an illustrative contract at
`evaluation_thresholds.example.json`. Apply it with:

```bat
python evaluate_replay_suite.py data/replay-suite ^
  --thresholds evaluation_thresholds.example.json ^
  --output reports/evaluation-suite.json
```

The report includes an `acceptance` section. A metric that has no independent
labelled samples is `NOT_MEASURED`, never a pass, and a failed threshold makes
the suite fail. These example values are not a claim that OptiVox has achieved
them; replace them with thresholds approved for the deployment.

Scenario files may declare normalized `tags` such as `known`, `unknown`,
`occlusion`, `liveness`, and `intrusion`. The acceptance gate can require
`coverage.required_tags` or `coverage.required_scenarios`, preventing a
promotion report from passing numeric checks while omitting a critical test
condition.

For liveness promotion, require the adversarial tags explicitly in the
threshold file rather than using a single generic `liveness` tag. A practical
site protocol includes `live_face`, `printed_photo`, `phone_screen`,
`replay_video`, `lighting_changes`, `glasses`, `mask`, `occlusion`, and
`multi_person`. These cases must be sourced from consented local recordings or
documented licensed data, with independent holdout/adversarial media kept out
of enrollment galleries.

The suite accepts one scenario, a JSON list, or an object containing a
`scenarios` list. Optional labels include `expected_name` and `expected_live`
on face records, `ground_truth_id` on tracks, `expected_security_events` at
scenario level, and `vehicle_speed_labels` with expected and predicted km/h.
Each case retains its own verdict and error. A policy `PASS` does not imply
model accuracy; accuracy remains `NOT_MEASURED` until independent holdout or
adversarial labels are supplied. Malformed cases fail the suite instead of
being silently skipped.

Recognition evaluation separates temporal `deferred` frames and
liveness-blocked frames from genuine false rejects. This prevents the
intentional candidate warm-up period from being counted as recognition
failure, while preserving the observed-frame count and those deferrals in the
report.

Embedding-gallery evaluation is available locally with
`evaluate_recognition.py`. Give it an original gallery and an independent
probe set; add `--augmented-gallery` to compare bounded augmentation. The
report contains FAR, FRR, unknown rejection, accuracy, and a conservative
recommendation. Embeddings are accepted only as local input and are never
written into the output report. For promotion evidence, use
`--require-independent-probes`; it requires gallery samples from `train` or
`validation` and probes from `holdout` or `adversarial`, each with a distinct
`source_id`:

```bat
python evaluate_recognition.py --gallery reports/original-gallery.json --augmented-gallery reports/augmented-gallery.json --probes reports/holdout-probes.json --require-independent-probes --output reports/recognition-evaluation.json
```

Age and crowd outputs are deliberately conservative. When age/gender inference
is enabled, the local overlay exposes an approximate age band such as `teen` or
`adult`, never an exact age. It is experimental and is not used for attendance,
access, discipline, or security decisions. Detector-based density remains an
experimental signal; perspective-correct density is explicitly
`NOT_CONFIGURED` until a calibrated camera and validation set exist.

## Dataset evaluation manifest

Use `build_dataset_manifest.py` to inventory local evaluation media without
copying it or assuming consent. The parent folder becomes the label, paths stay
relative to the dataset root, and the manifest records an explicit provenance
and split. Keep validation, holdout, and adversarial media in separate source
collections; do not place augmented derivatives in a different split from
their original source.

```bat
python build_dataset_manifest.py known_faces --output data/dataset-manifest.json --task face_verification --split train --provenance internal --require-files
python dataset_registry.py data/dataset-manifest.json --require-files
```

`internal` is the safe default because the tool cannot verify consent or a
third-party licence. Change it only when the dataset owner has documented the
provenance. Exact duplicate files, source split leakage, and checksum changes
are rejected by the registry; this is dataset hygiene, not a claim that the
model is accurate.

Security zones may optionally include `camera_id`. Zone transitions and
debounce/dwell state are scoped to that camera and track generation, preventing
one feed from inheriting another feed's intrusion history. The first observation
of a track in a zone establishes a baseline and does not fabricate an entry;
`ZONE_ENTRY` and `ZONE_INTRUSION` require a later outside-to-inside transition.
A zone without a `camera_id` remains applicable to the single active camera for
compatibility.

The protected `GET /api/system/security-zones` endpoint reports the active
policy, and `POST /api/system/security-zones` queues a validated replacement
for the local edge runtime. Replacement is atomic: malformed geometry,
duplicate IDs, invalid schedules, or invalid thresholds leave the previous
policy active. A successful replacement resets geometry-dependent intrusion,
dwell, PPE, and evacuation state, persists `runtime/security_zones.json`, and
adds an audit record. This is an operational policy control, not a substitute
for validating the physical camera placement and zone coordinates at the site.

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
events are recorded, and rejected if a checksum changes. When
`OPTIVOX_EVIDENCE_ENCRYPTION_ENABLED=true`, snapshot bytes are written as a
versioned AES-GCM envelope using the separate
`OPTIVOX_EVIDENCE_ENCRYPTION_KEY`; the API and alert senders decrypt only in
memory, so ordinary files in `snapshots/` are not viewable JPEGs. A bounded
`OPTIVOX_EVIDENCE_ENCRYPTION_KEYS_JSON` ring permits key rotation. Pilot and
production require this encryption policy and fail closed without the key;
development/exhibition may keep plaintext compatibility for existing local
snapshots. The protected evidence purge keeps paths referenced by incident
evidence and returns deleted, protected, and failed counts for the audit
record, so retention cannot silently break an incident timeline. Database backups are
created only through the protected System backup action and include a checksum
manifest. Evidence cleanup uses `OPTIVOX_EVIDENCE_RETENTION_DAYS` and should
be scheduled by the site operator after confirming the retention policy.
Backup manifests are checksum-protected in development and can be HMAC-signed
with `OPTIVOX_BACKUP_SIGNING_KEY`. Pilot and production modes require a
32-character-or-longer signing key and reject unsigned or tampered manifests;
the key must remain outside the repository and separate from the database. The
backup bytes are also AES-GCM encrypted when
`OPTIVOX_BACKUP_ENCRYPTION_ENABLED=true`, using the separate
`OPTIVOX_BACKUP_ENCRYPTION_KEY`. Pilot and production enable this policy by
default and fail closed without a key of at least 32 characters. Each manifest
stores only a non-secret `OPTIVOX_BACKUP_ENCRYPTION_KEY_ID`; during rotation,
keep the previous key in the bounded `OPTIVOX_BACKUP_ENCRYPTION_KEYS_JSON`
ring until its retention window expires. Encryption is applied before the
backup is atomically published; restore authenticates and decrypts into a
short-lived protected temporary file, then runs SQLite integrity and foreign-key
checks before replacement. The System storage status exposes a bounded backup
inventory with validity, encryption, and signature state, but never returns
local database paths or backup contents. Losing the encryption key makes those
backups unrecoverable, so it must be escrowed outside the repository.
Database health remains a lightweight request-path check. An authenticated
System administrator can explicitly run `/api/system/database/maintenance`
with full integrity checking, WAL checkpointing, and SQLite optimization; the
operation is recorded in the cybersecurity and audit histories.
Audit records in both SQLite adapters use a chained SHA-256 hash. The System
page exposes the current chain-integrity result; a failure means the audit
history requires investigation before it is trusted.
Platform audit actions also create signed checkpoints when
`OPTIVOX_AUDIT_SIGNING_KEY` is configured. Pilot and production modes require
that key, and integrity checks validate both the row chain and checkpoint
signatures. The latest signed checkpoint anchors the full verified chain, which
also gives existing development databases a controlled key-enablement path.
This makes unauthorized history rewriting detectable even when a malicious
actor can alter more than one audit row.

The database health report checks more than whether SQLite can open the file:
it runs `quick_check`, foreign-key validation, WAL mode reporting, migration
version reporting, and a required application-schema contract. A database can
therefore be marked `degraded` when it is physically readable but missing a
required table or column. It also checks semantic consistency, including
duplicate active presence sessions, unlinked incidents, and attendance
decisions without evidence. Related writes should use the explicit transaction
helpers so failures roll back as one unit. Backups are integrity-checked before
and after copying; restores require a matching versioned checksum manifest and
reject unverified backup files. Version 2 manifests also preserve the applied
edge/platform schema-state versions and checksums for restore verification.
Backend health also includes the platform audit-chain and checkpoint result, so
the database cannot report healthy while its audit history is known to be
corrupted.
The shared migration ledger also carries a checksum for each known migration;
editing a migration row directly is reported as a degraded database rather
than being treated as valid history.

Schema ownership is reported explicitly through `schema_migration_state`:
the `edge` and `platform` adapters record their applied version, description,
and contract checksum independently. This keeps the legacy audit migration
ledger compatible while making adapter drift visible in health diagnostics.
The legacy root `database.py` adapter remains supported for older modules, but
its initialization now invokes the same platform schema contract before it
returns, so it cannot silently create an incomplete database alongside the
canonical edge/API schema.
Health checks also compare those recorded states with the expected application
contract and degrade the database when a version or checksum is missing or
tampered with.
Each successful adapter application is also recorded in the append-only
`schema_migration_history` table with version, checksum, timestamp, process,
and host metadata. This is execution provenance, separate from the latest
state, and its update/delete guards are checked as part of database health.

The `outbox` state also stores a structural schema fingerprint derived from its
table and indexes. The platform state (version 17) fingerprints the operational,
cybersecurity, and edge-synchronization tables as a complete contract. Startup
health compares live SQLite DDL with the recorded fingerprint; an unapproved
index or table-shape change is reported as degraded instead of being silently
accepted. Deployment-specific SQLite defaults are excluded from the fingerprint
so edge and API adapters can retain local device defaults. A declared schema
upgrade deliberately refreshes the fingerprint only after its DDL is applied;
later unapproved changes remain visible as drift.

Deployment ownership fields are immutable after insertion. Scope-changing an
existing person, event, incident, or synchronization record is rejected at the
SQLite boundary, preventing a record from being silently re-parented while its
related evidence remains in the original site.

The database also rejects invalid operational states at write time: unsupported
identity, liveness, attendance, challenge, or incident states; negative
attendance durations; and timestamps where a clock-out or presence session
ends before it begins. This keeps impossible records out of downstream
analytics instead of relying only on application-layer validation.

Health checks also scan existing legacy/imported rows without mutating them.
They report impossible chronology, negative durations, unsupported states,
missing person references, duplicate open sessions, broken scope links, and
unlinked active incidents as explicit consistency issues. This distinguishes
preventive write protection from detection of historical corruption.

Audit hashes are versioned so schema evolution does not require rewriting
append-only history. Version 1 records use the original pre-scope payload,
version 2 records bind organization/site/device scope, and new version 3
records also bind the explicit hash-version marker. Health reports classify
legacy-compatible history and warn about weaker historical coverage instead of
silently presenting it as identical to current records.

SQLite durability is mode-aware. Development and exhibition default to
`synchronous=NORMAL` for a responsive local prototype; pilot and production
fail closed unless `OPTIVOX_SQLITE_SYNCHRONOUS=FULL`. Strict modes also enable
`secure_delete=ON`, reducing recovery of sensitive fragments from deleted
pages. These settings are applied by both the FastAPI database adapter and the
canonical local runtime adapter, so the two database owners do not silently
use different durability policies.

Version 4 added scoped composite indexes for roster lookup, attendance and
absence dates, event and audit timelines, incidents, cybersecurity alerts, and
edge synchronization checkpoints. These indexes are part of the fingerprinted
contract and are applied idempotently during startup.

Version 5 added SQLite write-time scope triggers for identity, attendance,
presence, evidence, incident, alert, and cybersecurity links. A cross-site
foreign reference now fails inside the transaction instead of merely being
reported later by health diagnostics; legacy corruption is still surfaced by
the consistency checker. Version 6 extended those guards to updates as well as
inserts, and included the trigger definitions in the structural fingerprint.
Version 7 adds a partial unique index for one active or occluded presence
session per scoped entity and camera. A legacy database with duplicate open
sessions fails this migration instead of silently accepting ambiguous
attendance state. Version 9 adds append-only SQLite guards for the edge and
platform audit tables and audit checkpoints; ordinary updates and deletes now
fail at the database boundary, while the hash-chain verifier remains the
recovery check for legacy or externally modified files. The audit writer's
single initialization update is explicitly allowed before a record becomes
immutable.

Version 10 extends the append-only boundary to attendance corrections,
incident review actions, and incident alert-delivery attempts. Corrections and
operator review history are never edited in place; a new correction or review
action must be appended, preserving the evidence trail used to explain an
official attendance or incident decision.

Version 11 adds an append-only migration-run ledger. Every platform schema
operation records a committed `started` marker before DDL and a terminal
`applied` or `failed` marker afterward. A crash or failed migration therefore
remains visible in health diagnostics instead of being indistinguishable from
a clean startup.

The outbox contract is version 3. Its structural fingerprint is scoped to the
outbox table and its own indexes/triggers, so unrelated platform protections
cannot create a false outbox-drift alarm.

Schema initialization is serialized with a shared `<database>.schema.lock`
sidecar understood by both the local edge adapter and the FastAPI adapter.
This protects multi-statement additive migrations when the camera runtime and
backend start at the same time. Health checks also verify the required
uniqueness indexes and scope/audit triggers, so a missing protection object
is reported as degraded rather than silently accepted. Database health exposes
capacity guardrails from `OPTIVOX_DATABASE_MAX_BYTES`,
`OPTIVOX_DATABASE_WARN_BYTES`, `OPTIVOX_DATABASE_WAL_MAX_BYTES`, and
`OPTIVOX_DATABASE_MIN_FREE_BYTES`; exceeding them produces an operational
warning or critical state before disk exhaustion becomes a data-loss event.
It also exposes bounded process-local database metrics: connection and
transaction counts, commit/rollback/error totals, lock contention, and p50/p95
connection and transaction latency. These metrics are diagnostic samples, not
a substitute for the controlled multi-minute benchmark required before a
production capacity claim.
An authenticated system administrator can request schema repair through the
database maintenance action; it reuses the migration lock, reapplies the
idempotent contract, and records the resulting health state instead of hiding
the repair.

Database and storage health also expose a bounded backup-recovery status. It
reports whether a recent valid, policy-compliant backup exists, including its
age and invalid-backup count, without exposing local filesystem paths. Set
`OPTIVOX_BACKUP_MAX_AGE_MINUTES` to the maximum acceptable recovery-point age.
The public system-status API similarly redacts the local database path while
retaining safe health, capacity, migration, backup, and latency metadata.
Pilot and production startup also require either the built-in scheduler or an
explicit `OPTIVOX_EXTERNAL_BACKUP_CONFIRMED=true` declaration for an external
backup owner. Development and exhibition modes surface missing recovery points
as warnings so local experimentation remains usable.

The live operational database uses standard SQLite in development/exhibition.
Pilot/production configuration requires the optional SQLCipher-compatible
connection boundary: install `pysqlcipher3` or `sqlcipher3`, set
`OPTIVOX_DATABASE_ENCRYPTION_ENABLED=true`, and provide a 32-character-or-
longer `OPTIVOX_DATABASE_ENCRYPTION_KEY` outside the repository. The edge and
FastAPI adapters, backups, restores, and legacy adapter all use that boundary
when enabled and fail closed if the driver is missing. Embeddings, evidence
snapshots, and backups retain their separate authenticated storage controls.
`secure_delete` alone is not equivalent to full-database encryption, and the
application does not claim production at-rest protection until the SQLCipher
or OS-level disk-encryption gate is actually configured and tested.

Operational rows carry `organization_id`, `site_id`, and `device_id` scope
metadata. The backend applies that boundary to roster, attendance, academic,
analytics, event, incident, and profile queries, so a numeric record ID cannot
silently expose another site's data. Existing local databases are backfilled
with the configured local scope during startup; changing scope requires an
intentional deployment configuration change and a migration/backup review.

Automatic attendance follows the configured school calendar. By default,
Monday through Friday are school days. Override this for a pilot site with
`OPTIVOX_SCHOOL_DAYS=0,1,2,3,4` (Monday is `0`, Sunday is `6`) and optionally provide a
comma-separated holiday list with `OPTIVOX_SCHOOL_HOLIDAYS=2026-10-01,2026-12-25`.
Non-school days skip automatic attendance and are not treated as inferred
absence days by the academic overview.

## Edge synchronization boundary

The local runtime also maintains `runtime/sync_outbox.ndjson`. It is an
append-only, idempotent, hash-chained queue for minimized operational events
that a future control-plane connector can upload after device authentication.
Embeddings, raw frames, local plate text, credentials, and tokens are refused
by the outbox. Without `OPTIVOX_SYNC_SECRET` it remains explicitly
`LOCAL_ONLY_UNSIGNED`; setting that secret enables HMAC signatures but does
not perform network access by itself. Set `OPTIVOX_DEVICE_ID`,
`OPTIVOX_SITE_ID`, and `OPTIVOX_ORGANIZATION_ID` to bind a queue to one edge
device and deployment scope. A future connector must validate a contiguous
sequence, device/site/organization identity, hash chain, HMAC signature, and
last accepted checkpoint before applying a batch. Attendance decisions are
also queued as minimized provenance records without names or biometric
material.

An explicit synchronization worker is available for a controlled pilot. Set
`OPTIVOX_SYNC_ENABLED=true`, `OPTIVOX_SYNC_ENDPOINT=https://...`,
`OPTIVOX_SYNC_SECRET` (at least 32 characters), and
`OPTIVOX_SYNC_ALLOWED_HOSTS` before starting it. The endpoint must be HTTPS,
credential-free, and allowlisted. The worker is isolated from capture,
inference, rendering, and attendance; a slow receiver cannot freeze local
operation. It sends contiguous batches and acknowledges them only when the
receiver returns the expected sequence and hash. The FastAPI receiver at
`POST /api/sync/ingest` stores an idempotent `edge_sync_batches` receipt,
`edge_sync_events`, and an `edge_sync_devices` checkpoint. This first receiver
accepts the configured local device by default. For a controlled multi-device
pilot, `OPTIVOX_SYNC_DEVICE_REGISTRY_JSON` can define active devices with
separate secrets and organization/site scopes; secrets remain environment
material and are never stored in or returned by the database. Revoked or
malformed entries are rejected at configuration validation. The registry can
optionally require Ed25519 request signatures by supplying `signature_algorithm`,
`key_id`, and an Ed25519 `public_key`; the edge keeps its matching private key
at `OPTIVOX_SYNC_PRIVATE_KEY_PATH`. Record-chain HMAC remains required for
compatibility and defense in depth. Key rotation is performed by replacing the
registered public key and edge private-key configuration during a controlled
deployment; a full administrator-managed provisioning service is still future
work.

For an Ed25519 device, generate a keypair outside Git, keep the private PEM
under the local project deployment root, and register only the public PEM:

```python
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

key = Ed25519PrivateKey.generate()
print(key.private_bytes(serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode())
print(key.public_key().public_bytes(serialization.Encoding.PEM,
    serialization.PublicFormat.SubjectPublicKeyInfo).decode())
```

The receiver registry entry must include `signature_algorithm: "ed25519"`, a
stable `key_id`, the public PEM, and the existing per-device `secret` used to
verify the append-only event chain. Never paste the private PEM into the
registry, frontend, logs, or a committed `.env` file.
An authenticated system administrator can also use
`POST /api/sync/devices/{device_id}/status` to revoke or reactivate a
registered device. Revocation preserves its last checkpoint for audit and
recovery but blocks both new batches and replayed deliveries.
Correlated security envelopes also carry organization, site, device, camera,
entity, and presence context before persistence, so a future control plane can
enforce scope at the event boundary rather than relying only on a dashboard
filter.

The backend also has a transactional `platform_outbox` table. Attendance,
profile, incident, and cybersecurity mutations enqueue minimized operational
events in the same SQLite transaction as the source change, so a committed
record cannot be silently lost between persistence and future synchronization.
Events are idempotent, scoped to organization/site/device, checksum-protected,
claimable by one worker at a time, retryable with exponential backoff, and
ultimately moved to `dead_letter` after repeated failure. Payloads containing
embeddings, credentials, raw frames, or local plate text are rejected. The
System database status reports pending, sent, failed, and dead-letter counts;
the outbox is a durable hand-off mechanism, not a network sender.
Abandoned `processing` claims are returned to retryable state after a bounded
worker lease expires, so a crashed sender cannot strand operational events
forever.
Dead-letter items can be requeued only through the protected System operation,
and sent-item retention is previewable before deletion. Authentication session
creation, rotation, revocation, and audit records also share one transaction;
an audit failure cannot leave a half-created login session.

The protected `POST /api/system/database/retention` operation provides a
dry-run by default and can explicitly purge only unlinked low-level telemetry
after operator confirmation. Attendance, roster, incident, evidence metadata,
and audit history are protected. The purge is transactional and creates an
audit record, so a retention action is visible and reversible through backup
recovery rather than being an invisible scheduled deletion.

Database backup retention is also explicit and conservative. The protected
`POST /api/system/backups/retention` operation previews deletion first, keeps a
minimum newest-backup floor, removes only checksum- and signature-verified
backup/manifest pairs older than the configured window, and preserves invalid
backups for investigation. An optional isolated scheduler can be enabled with
`OPTIVOX_BACKUP_SCHEDULER_ENABLED=true`; it runs outside request, camera, and
sync workers and reports success, failure, last backup, and cleanup state in
the System view. The scheduler is deliberately disabled by default so a site
must choose and test its recovery policy before automatic deletion begins.

Administrators can run `POST /api/system/backups/recovery-drill` to restore the
newest verified backup into an isolated temporary database. The drill checks
SQLite integrity and foreign keys, deletes its temporary materialization, never
replaces the live database, and records an audit/cybersecurity outcome.
