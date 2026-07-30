@echo off
setlocal
cd /d "%~dp0"
"C:\Users\digit\anaconda3\envs\Pytorch\python.exe" demo.py --config profiles\j_run12_predicted_actual_seed1\config.json
if errorlevel 1 pause
