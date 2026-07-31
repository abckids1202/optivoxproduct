# OptiVox Web

OptiVox is now split into two deployable parts:

- `frontend/`: Vite + React browser app.
- `backend/`: Flask JSON API, authentication, live MJPEG stream, and OptiVox vision bridge.

## Run Locally

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## API Health

The backend exposes:

```text
GET http://localhost:8000/api/v1/health
```

Expected response:

```json
{
  "status": "healthy",
  "service": "optivox-backend"
}
```

## Deploy

For static shared hosting such as Hostinger, build the frontend:

```powershell
cd frontend
npm install
npm run build
```

Upload only the contents of `frontend/dist/` to `public_html/`.

The Python backend cannot run on static/shared hosting. Host `backend/` on a Python-capable service or VPS and set `frontend/.env`:

```text
VITE_API_BASE_URL=https://your-backend-domain.example
```

Do not upload `.env`, backend source, database files, model secrets, or Python files into `public_html/`.
