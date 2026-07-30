@echo off
setlocal
cd /d "%~dp0"
"C:\Users\digit\anaconda3\envs\Pytorch\python.exe" demo.py --validate --config config.json
if errorlevel 1 goto failed
"C:\Users\digit\anaconda3\envs\Pytorch\python.exe" demo.py --validate --config profiles\a_run7_predicted_actual_seed1\config.json
if errorlevel 1 goto failed
"C:\Users\digit\anaconda3\envs\Pytorch\python.exe" demo.py --validate --config profiles\j_run12_seed1\config.json
if errorlevel 1 goto failed
"C:\Users\digit\anaconda3\envs\Pytorch\python.exe" demo.py --validate --config profiles\j_run12_predicted_actual_seed1\config.json
if errorlevel 1 goto failed
echo.
echo All four demo profiles passed validation.
pause
exit /b 0
:failed
echo.
echo Validation failed.
pause
exit /b 1
