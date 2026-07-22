@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
"%PYTHON_BIN%" tools\validate_setup.py ^
  --dataset-root "%DATASET_ROOT%" ^
  --task-graph "%TASK_GRAPH%" ^
  --relation-matrix "%RELATION_MATRIX%" ^
  --camera-id "%CAMERA_ID%" ^
  --test-participant "%TEST_PARTICIPANT%"
exit /b %ERRORLEVEL%
