@echo off
cd /d "%~dp0"
start "OptiVox Engine" cmd /k start_optivox.bat
start "OptiVox Backend" cmd /k start_backend.bat
start "OptiVox Frontend" cmd /k start_frontend.bat
