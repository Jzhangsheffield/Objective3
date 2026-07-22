@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
"%PYTHON_BIN%" tools\summarize_all_models.py ^
  --outputs-root "%OUTPUTS_ROOT%" ^
  --output-dir "%TRAIN_SCOPE_COMPARISON_CROSS_ROOT%" ^
  --participants A D M ^
  --train-scopes normal_only all_runs ^
  --representation-scopes normal_only all_runs ^
  --matched-scope-only
exit /b %ERRORLEVEL%
