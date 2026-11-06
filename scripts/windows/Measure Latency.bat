@echo off
rem Double-click, then clap slowly 6 times with both hands in view of the camera.
rem The microphone hears each clap instantly; the camera sees it later. The difference is the camera delay.
title Latency test
cd /d "%~dp0..\.."
".venv\Scripts\python.exe" app.py studio --latency-test %*
pause
