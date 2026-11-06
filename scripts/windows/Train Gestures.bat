@echo off
rem Double-click to record a new hand gesture and retrain the gesture model.
title Train Gestures
cd /d "%~dp0..\.."
".venv\Scripts\python.exe" app.py gestures list
echo.
echo Type a gesture name to record it (e.g. peace, thumbs_up, none), or just press Enter to train.
:loop
set "name="
set /p "name=Gesture name (Enter = train and finish): "
if "%name%"=="" goto train
".venv\Scripts\python.exe" app.py gestures record %name%
goto loop
:train
".venv\Scripts\python.exe" app.py gestures train
pause
