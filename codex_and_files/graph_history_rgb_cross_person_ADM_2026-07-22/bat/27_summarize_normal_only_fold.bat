@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
"%PYTHON_BIN%" tools\summarize_all_models.py ^
  --outputs-root "%OUTPUTS_ROOT%" ^
  --output-dir "%NORMAL_FOLD_SUMMARY_ROOT%" ^
  --participants "%TEST_PARTICIPANT%" ^
  --seeds "%SEED%" ^
  --train-scopes normal_only ^
  --representation-scopes normal_only ^
  --matched-scope-only ^
  --require-complete-grid
exit /b %ERRORLEVEL%
