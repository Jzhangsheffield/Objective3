@echo off
setlocal
set "EXPERIMENTS=%~1"
if not defined EXPERIMENTS set "EXPERIMENTS=A0,A1,A2,A3,A4,A5,A6,A7,A8"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_experiments.ps1" -Experiments "%EXPERIMENTS%"
exit /b %ERRORLEVEL%

