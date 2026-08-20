@echo off
setlocal
set "EXPERIMENTS=%~1"
if not defined EXPERIMENTS set "EXPERIMENTS=A3-DualPos,A4-DualPos"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_experiments.ps1" -Experiments "%EXPERIMENTS%"
exit /b %ERRORLEVEL%
