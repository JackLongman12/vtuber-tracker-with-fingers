@echo off
rem Double-click for the launcher window: camera / microphone choice, what's set up, buttons for every tool.
cd /d "%~dp0..\.."
start "" ".venv\Scripts\pythonw.exe" app.py
