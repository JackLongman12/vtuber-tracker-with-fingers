@echo off
rem Double-click to start the avatar studio (webcam -> VTuber app + 3D viewer).
rem Extra options can be added after app.py studio, e.g.  app.py studio --mirror-avatar --expressiveness 1.3
title Avatar Studio
cd /d "%~dp0..\.."
echo Starting Avatar Studio... (first start takes ~10 seconds)
echo Settings:  http://127.0.0.1:8000/settings.html
echo 3D viewer: http://127.0.0.1:8000    VMC: 127.0.0.1:39539
echo In the studio window: C calibrate, V vowels, 1-9 expressions, F face monitor, Q quit
echo Anywhere: Ctrl+Alt+1-9 toggles expressions
echo.
".venv\Scripts\python.exe" app.py studio %*
if errorlevel 1 (
    echo.
    echo Avatar Studio stopped with an error - see the messages above.
    pause
)
