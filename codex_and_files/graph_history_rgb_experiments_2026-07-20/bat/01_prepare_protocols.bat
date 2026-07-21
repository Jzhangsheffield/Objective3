@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
"%PYTHON_BIN%" tools\prepare_protocols.py ^
  --dataset-root "%DATASET_ROOT%" ^
  --output-root "%PROTOCOL_ROOT%" ^
  --test-participant J
exit /b %ERRORLEVEL%

