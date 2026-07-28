@echo off
setlocal EnableExtensions
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
"%PYTHON_BIN%" tools\summarize_direct_head_fusion.py ^
  --outputs-root "%OUTPUTS_ROOT%" ^
  --output-dir "%DIRECT_FOURFOLD_SUMMARY_ROOT%" ^
  --camera-id "%CAMERA_ID%" ^
  --participants A D J M ^
  --seeds 1 2 42 ^
  --train-scopes normal_only all_runs ^
  --require-complete-grid
exit /b %ERRORLEVEL%
