# OptiVox Migration Map

The original Flask monolith was separated into a deployable static frontend and a Python backend.

## Frontend

- `templates/base.html` -> `frontend/src/App.jsx` layout, nav, and footer components.
- `templates/index.html` -> `frontend/src/App.jsx` `HomePage`.
- `templates/features.html` -> `frontend/src/App.jsx` `FeaturesPage`.
- `templates/demo.html` + `static/js/demo.js` -> `frontend/src/App.jsx` `DemoPage`.
- `templates/about.html` -> `frontend/src/App.jsx` `AboutPage`.
- `templates/faq.html` + FAQ script behavior -> `frontend/src/App.jsx` `FaqPage`.
- `templates/contact.html` -> `frontend/src/App.jsx` `ContactPage`.
- `templates/login.html` -> `frontend/src/App.jsx` `LoginPage`.
- `templates/dashboard/index.html` + `static/js/dashboard.js` -> `frontend/src/App.jsx` `DashboardPage`.
- `static/css/optivox.css` -> `frontend/src/index.css`.
- `static/img/optivox-logo.jpeg` -> `frontend/public/images/optivox-logo.jpeg`.

## Backend

- `app.py` -> `backend/app/main.py`.
- `config.py` -> `backend/app/config.py`.
- `run.py` -> `backend/run.py`.
- `requirements.txt` -> `backend/requirements.txt`.
- `core/` -> `backend/app/core/`.
- `blueprints/auth.py` -> `backend/app/api/routes/auth.py`.
- `blueprints/api.py` -> `backend/app/api/routes/vision.py`.
- Template-rendered routes were replaced by the React frontend.

## API Prefix

React uses `/api/v1`. Legacy `/api` aliases remain registered for compatibility.
