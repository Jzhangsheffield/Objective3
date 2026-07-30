@echo off
setlocal EnableExtensions
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
if exist "%ATOMIC_TAIL_FOURFOLD_SUMMARY_ROOT%\completed.json" (
  echo Completed atomic-tail summary exists; skip without overwrite.
  exit /b 0
)
"%PYTHON_BIN%" tools\summarize_atomic_tail_graph_valid.py ^
  --outputs-root "%OUTPUTS_ROOT%" ^
  --output-dir "%ATOMIC_TAIL_FOURFOLD_SUMMARY_ROOT%" ^
  --camera-id "%CAMERA_ID%" ^
  --participants A D J M ^
  --seeds 1 2 42 ^
  --train-scopes normal_only all_runs ^
  --refresh-policies refresh_every_1 refresh_every_10 refresh_once ^
  --require-complete-grid
exit /b %ERRORLEVEL%
