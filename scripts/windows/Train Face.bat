@echo off
rem Double-click: about 2 minutes of guided expressions (smile, brows up, wink, tongue out, ...). Afterwards the
rem tracker knows how your face reads and corrects it, and can show tongue out / cheek puff. Face the camera in
rem good light. Close the studio first.
title Train personal face refiner
cd /d "%~dp0..\.."
".venv\Scripts\python.exe" app.py train-face %*
pause
