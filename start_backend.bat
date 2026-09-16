@echo off
cd /d "%~dp0"
if /I "%OPTIVOX_RUNTIME_MODE%"=="production" goto production
if /I "%OPTIVOX_RUNTIME_MODE%"=="pilot" goto production
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
goto end
:production
python -m uvicorn backend.main:app --host %OPTIVOX_HOST% --port %OPTIVOX_PORT%
:end
