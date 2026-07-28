@echo off
setlocal
cd /d "%~dp0"
"C:\Users\digit\anaconda3\envs\Pytorch\python.exe" build_display_video.py --config config.json
if errorlevel 1 goto failed
"C:\Users\digit\anaconda3\envs\Pytorch\python.exe" build_display_video.py --config profiles\j_run12_seed1\config.json
if errorlevel 1 goto failed
echo.
echo Both display videos were built and validated.
pause
exit /b 0
:failed
echo.
echo Display video build failed.
pause
exit /b 1
