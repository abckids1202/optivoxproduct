# OptiVox — Deployment Guide

This guide takes the platform from a laptop demo to a reachable production
deployment. Read `ARCHITECTURE.md` first — the deployment choices below are
shaped by one hard constraint: **inference runs once, in a single background
worker.** That single fact drives the worker count, the proxy config, and the
scaling path.

---

## 0. The one rule that shapes everything

OptiVox does heavy CV work (YOLOv8 + InsightFace + MediaPipe) on every frame.
That pipeline runs **exactly once** inside `VisionBridge`, which lives in the
Flask process. Browsers never trigger inference — they pull pre-rendered JPEG
frames over MJPEG. Consequence:

> **Run a single application process with a single worker.** Multiple workers =
> multiple camera grabs + multiple model loads = GPU/CPU contention and a
> doubled or tripled compute bill for zero benefit.

If you outgrow one machine, you do **not** add workers — you split the vision
worker into its own service (see "Scaling out" below).

---

## 1. Local run (development)

```bash
cd optivox-web
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then edit .env
python run.py
```

Open `http://localhost:8000`. Dashboard login (demo): `admin` / `optivox`.

By default the bridge runs in **synthetic mode** (an animated placeholder frame)
so the whole site works before your CV stack is wired in. Confirm with:

```bash
curl -s http://localhost:8000/api/health
# {"status":"ok","bridge":"running","mode":"synthetic", ...}
```

---

## 2. Connecting your real CV system

| Step | Action |
|------|--------|
| 1 | Save your single-file system as `optivox_core.py` next to `run.py` (or anywhere on `PYTHONPATH`). |
| 2 | Make sure it exposes `VisionSystem`, `EventDatabase`, `AttendanceManager` at module top level (it already does). |
| 3 | Remove or guard the `if __name__ == "__main__": main()` call — the bridge imports your classes, it does **not** run your `main()` loop (no `cv2.imshow`). |
| 4 | Install the heavy deps (uncomment them in `requirements.txt`): `ultralytics`, `insightface`, `onnxruntime`, `mediapipe`, `faiss-cpu`, `torch`. |
| 5 | Set env vars (below) and restart. |

```ini
# .env
OPTIVOX_VISION_MODULE=optivox_core
OPTIVOX_CAMERA=0                 # webcam index, or rtsp://user:pass@cam-ip:554/stream
OPTIVOX_DB=../security.db        # path to the SQLite file your core writes to
OPTIVOX_ENABLE_BRIDGE=1
OPTIVOX_SYNTHETIC=1        # set 0 in prod to fail loudly if camera/module missing
OPENAI_API_KEY=sk-...            # optional; only for the AI assistant
OPTIVOX_SECRET_KEY=change-me-long-random
```

Health check should now report `"mode":"live"`. If it still says `synthetic`,
read the bridge log line — it states exactly why (module not importable, camera
not opening, etc.).

> ⚠️ **Security:** your pasted core had a hard-coded OpenAI key. Rotate it now
> and only ever supply keys through `OPENAI_API_KEY`. Never commit `.env`.

---

## 3. Production on a single machine (recommended start)

Gunicorn, **one worker, threaded** — the threads serve many MJPEG viewers; the
single worker guarantees one vision pipeline.

```bash
pip install gunicorn
gunicorn --workers 1 --threads 8 --timeout 120 --bind 127.0.0.1:8000 app:app
```

- `--workers 1` — non-negotiable (see §0).
- `--threads 8` — concurrent viewers / API calls. Raise for more dashboard seats.
- `--timeout 120` — MJPEG is a long-lived response; don't let gunicorn kill it.
- `app:app` — the factory in `app.py` starts the bridge on import.

### Put a reverse proxy in front (nginx)

TLS termination, buffering off for the stream, and a stable public surface:

```nginx
server {
    listen 443 ssl;
    server_name optivox.yourdomain.com;
    ssl_certificate     /etc/letsencrypt/live/optivox.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/optivox.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # MJPEG stream: disable buffering so frames flush immediately
    location /api/video_feed {
        proxy_pass http://127.0.0.1:8000;
        proxy_buffering off;
        proxy_read_timeout 3600s;
        chunked_transfer_encoding on;
    }
}
```

Get the cert with `certbot --nginx -d optivox.yourdomain.com`.

### Keep it alive (systemd)

```ini
# /etc/systemd/system/optivox.service
[Unit]
Description=OptiVox
After=network.target

[Service]
WorkingDirectory=/opt/optivox-web
EnvironmentFile=/opt/optivox-web/.env
ExecStart=/opt/optivox-web/.venv/bin/gunicorn --workers 1 --threads 8 --timeout 120 --bind 127.0.0.1:8000 app:app
Restart=always
User=optivox

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now optivox && sudo systemctl status optivox
```

---

## 4. Remote access without a public server

The camera is on a machine inside your network. Two ways to reach it remotely
without renting a VPS:

| Method | Command | Notes |
|--------|---------|-------|
| **Cloudflare Tunnel** | `cloudflared tunnel --url http://localhost:8000` | Free, gives an HTTPS URL, no port-forwarding, survives NAT. Best default. |
| **ngrok** | `ngrok http 8000` | Fastest for a quick share/demo; free tier rotates the URL. |

Both keep all inference **local** — only the rendered stream + API JSON leave
the box. Put the dashboard behind login (already enforced) before exposing it.

---

## 5. Where the camera physically lives

```
[ Camera/NVR ] --RTSP--> [ OptiVox machine: Flask + VisionBridge ]
                              | renders annotated MJPEG
                              v
              [ nginx/TLS or Cloudflare Tunnel ]
                              |
                              v
                     [ Remote dashboard viewers ]
```

- **USB webcam** → `OPTIVOX_CAMERA=0`. Must run on the machine the cam is plugged into.
- **IP camera / NVR** → `OPTIVOX_CAMERA=rtsp://...`. OptiVox can run on any box that can reach the camera's RTSP URL — even a different machine on the LAN.
- **Raspberry Pi** → works for capture + light models; offload YOLO/InsightFace to a GPU box and point the Pi's stream at it (edge-capture, central-inference) when you need full detection.

---

## 6. Hardening checklist (before exposing publicly)

- [ ] `OPTIVOX_SECRET_KEY` set to a long random value; `.env` not committed.
- [ ] Replace the demo `admin/optivox` login with real per-org credentials (hash + store; see `blueprints/auth.py`).
- [ ] TLS on (nginx + certbot, or the tunnel's HTTPS).
- [ ] `OPTIVOX_SYNTHETIC=0` so a broken camera fails loudly instead of silently serving a fake feed.
- [ ] Rate-limit `/api/contact` and `/login` (nginx `limit_req` or `flask-limiter`).
- [ ] Rotate any leaked API keys; supply secrets only via env.
- [ ] Back up `security.db` (it holds people, events, attendance).
- [ ] Confirm the dashboard 302-redirects to `/login` when logged out (it does).

---

## 7. Scaling out (when one machine isn't enough)

You don't scale by adding web workers — you separate concerns:

| Stage | Change |
|-------|--------|
| **More dashboard viewers** | Raise gunicorn `--threads`; MJPEG is cheap to fan out. |
| **More cameras** | Run one OptiVox vision worker **per camera/site**; aggregate their event APIs behind one front-end. |
| **Heavier models / GPU** | Split `VisionBridge` into a standalone inference service (its own process/box with the GPU); have the web app read the shared frame buffer + DB. This is the documented monolith→service refactor in `ARCHITECTURE.md`. |
| **More write throughput** | Migrate SQLite → PostgreSQL (`core/db.py` is the only file that needs the DSN swap). |
| **Frame buffer / pub-sub across boxes** | Introduce Redis for the latest-frame handoff and event fan-out. |
| **Many orgs (SaaS)** | Containerize (Docker), one stack per tenant, orchestrate with Kubernetes; PostgreSQL with row-level tenant isolation. |

The key property: every stage above is additive. Nothing in the current
structure has to be rewritten — `core/` already isolates the bridge, the stream
buffer, and the DB accessor so each can move to its own service independently.

---

## 8. Common issues

| Symptom | Cause / Fix |
|---------|-------------|
| Dashboard shows the synthetic placeholder | `mode:synthetic` — check `/api/health` log line; module not importable or camera not opening. |
| Stream freezes after ~30s behind nginx | `proxy_buffering off` + long `proxy_read_timeout` missing on `/api/video_feed`. |
| High CPU, low FPS | Expected on CPU-only with full models. Raise `*_EVERY_N_FRAMES` in your core config, or move to GPU. |
| Two camera windows / double model load | You ran >1 gunicorn worker. Set `--workers 1`. |
| 500 on `/api/events` | DB path wrong (`OPTIVOX_DB`) or core hasn't created the schema yet — run your core once to initialise `security.db`. |
