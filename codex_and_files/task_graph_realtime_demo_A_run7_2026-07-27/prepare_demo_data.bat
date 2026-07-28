@echo off
setlocal
cd /d "%~dp0"
"C:\Users\digit\anaconda3\envs\Pytorch\python.exe" prepare_demo_metadata.py --config config.json
if errorlevel 1 goto failed
"C:\Users\digit\anaconda3\envs\Pytorch\python.exe" prepare_demo_metadata.py --config profiles\j_run12_seed1\config.json
if errorlevel 1 goto failed
echo.
echo Both demo profiles were prepared and validated.
pause
exit /b 0
:failed
echo.
echo Demo metadata preparation failed.
pause
exit /b 1
