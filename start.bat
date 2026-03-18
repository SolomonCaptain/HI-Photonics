@echo off
echo ============================================================
echo HI-Photonics Studio - Starting Development Environment
echo ============================================================

echo.
echo [1/2] Starting API Server...
start "HI-Photonics API" cmd /k "cd /d %~dp0api && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak > nul

echo.
echo [2/2] Starting Frontend Development Server...
start "HI-Photonics UI" cmd /k "cd /d %~dp0frontend && npm start"

echo.
echo ============================================================
echo Development environment started!
echo.
echo API Server: http://localhost:8000
echo API Docs:   http://localhost:8000/docs
echo Frontend:   http://localhost:3000
echo ============================================================
pause
