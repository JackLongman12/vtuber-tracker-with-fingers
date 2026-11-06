@echo off
rem Double-click to send a fixed test sequence of poses, faces and vowels to your VTuber app (no camera needed).
rem Compare your avatar with the description shown for each step; a wrong step shows which mapping is off.
title Avatar test pattern
cd /d "%~dp0..\.."
echo Make sure your VTuber app is receiving VMC on port 39539, then watch your avatar.
echo The 3D viewer (http://127.0.0.1:8000) shows each step's description too.
".venv\Scripts\python.exe" app.py studio --test-pattern %*
