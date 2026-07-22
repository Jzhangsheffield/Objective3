@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
"%PYTHON_BIN%" tools\summarize_all_models.py ^
  --outputs-root "%OUTPUTS_ROOT%" ^
  --output-dir "%UNIFIED_FOLD_SUMMARY_ROOT%" ^
  --participants "%TEST_PARTICIPANT%" ^
  --seeds "%SEED%"
exit /b %ERRORLEVEL%

