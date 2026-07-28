@echo off
setlocal
cd /d "%~dp0"
"C:\Users\digit\anaconda3\envs\Pytorch\python.exe" demo.py --config config.json
