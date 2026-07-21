@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
"%PYTHON_BIN%" tools\summarize_results.py ^
  --results-root "%MODEL_ROOT%" ^
  --output "%MODEL_ROOT%\experiment_summary.csv"
exit /b %ERRORLEVEL%

