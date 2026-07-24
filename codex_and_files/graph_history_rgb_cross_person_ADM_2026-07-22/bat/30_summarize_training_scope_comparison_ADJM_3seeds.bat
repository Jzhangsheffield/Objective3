@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
"%PYTHON_BIN%" tools\summarize_all_models.py ^
  --outputs-root "%OUTPUTS_ROOT%" ^
  --output-dir "%FOURFOLD_COMPARISON_ROOT%" ^
  --participants A D J M ^
  --seeds 1 2 42 ^
  --train-scopes normal_only all_runs ^
  --representation-scopes normal_only all_runs ^
  --matched-scope-only ^
  --require-complete-grid
exit /b %ERRORLEVEL%
