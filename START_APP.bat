@echo off
setlocal
cd /d "%~dp0"
title Customer Experience Decision Support - Start
if "%APP_PORT%"=="" set APP_PORT=8502

echo.
echo  ============================================================
echo   Multimodal Customer Experience Decision Support
echo  ============================================================
echo.

where docker >nul 2>&1
if errorlevel 1 (
  echo  Docker is not installed on this laptop.
  echo  Please install Docker Desktop first:
  echo  https://www.docker.com/products/docker-desktop/
  echo.
  pause
  exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
  echo  Docker Desktop is not running yet.
  echo.
  echo    1. Open Docker Desktop from the Start menu.
  echo    2. Wait until it says it is running.
  echo    3. Double-click START_APP.bat again.
  echo.
  pause
  exit /b 1
)

echo  Starting the app. Please wait and do not close this window.
echo  The FIRST time it downloads and prepares everything. This can
echo  take 10 to 20 minutes and needs an internet connection.
echo  Later starts take under a minute.
echo.
docker compose up -d --build
if errorlevel 1 (
  echo.
  echo  The app could not be started. Please read the message above.
  echo  If it says the port is already allocated, another program is
  echo  using port %APP_PORT%. Close that program, or open a Command
  echo  Prompt here and run:  set APP_PORT=8600  then START_APP.bat
  echo.
  pause
  exit /b 1
)

echo.
echo  Waiting for the app to get ready. If the models have to be
echo  trained first, this can take a few more minutes...
powershell -NoProfile -Command "$u='http://localhost:%APP_PORT%/_stcore/health'; for($i=0;$i -lt 400;$i++){ try { if((Invoke-WebRequest -UseBasicParsing $u -TimeoutSec 3).StatusCode -eq 200){ exit 0 } } catch {}; Start-Sleep 3 }; exit 1"
if errorlevel 1 (
  echo.
  echo  The app did not become ready in time. To see what is happening,
  echo  open Docker Desktop, click Containers, then click this app.
  echo.
  pause
  exit /b 1
)

echo.
echo  READY. The app is at  http://localhost:%APP_PORT%
echo  In the left sidebar, click "Live Triage" to see the AI agent.
echo  To stop the app later, double-click STOP_APP.bat
echo.
if not defined NO_BROWSER start "" http://localhost:%APP_PORT%
pause
