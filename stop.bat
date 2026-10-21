@echo off
setlocal
cd /d "%~dp0"
title YOLO26 Stop
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1"
if errorlevel 1 (
  echo.
  pause
) else (
  ping -n 4 127.0.0.1 >nul 2>&1
)
