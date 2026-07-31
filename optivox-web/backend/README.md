# OptiVox Backend

Flask JSON API for OptiVox authentication, dashboard data, live MJPEG video, and vision bridge control.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

The backend listens on port `8000` by default and exposes `/api/v1/health`.
