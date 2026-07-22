@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
"%PYTHON_BIN%" tools\summarize_all_models.py ^
  --outputs-root "%OUTPUTS_ROOT%" ^
  --output-dir "%ALLRUN_CROSS_SUMMARY_ROOT%" ^
  --participants A D M ^
  --train-scopes all_runs ^
  --representation-scopes all_runs ^
  --matched-scope-only
exit /b %ERRORLEVEL%
