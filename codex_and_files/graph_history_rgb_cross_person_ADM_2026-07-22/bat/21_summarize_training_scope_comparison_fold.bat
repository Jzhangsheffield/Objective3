@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
"%PYTHON_BIN%" tools\summarize_all_models.py ^
  --outputs-root "%OUTPUTS_ROOT%" ^
  --output-dir "%TRAIN_SCOPE_COMPARISON_FOLD_ROOT%" ^
  --participants "%TEST_PARTICIPANT%" ^
  --seeds "%SEED%" ^
  --train-scopes normal_only all_runs ^
  --representation-scopes normal_only all_runs ^
  --matched-scope-only
exit /b %ERRORLEVEL%
