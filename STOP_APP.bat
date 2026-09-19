@echo off
cd /d "%~dp0"
title Customer Experience Decision Support - Stop
echo Stopping the app...
docker compose down
echo.
echo The app has been stopped. Double-click START_APP.bat to run it again.
echo.
pause
