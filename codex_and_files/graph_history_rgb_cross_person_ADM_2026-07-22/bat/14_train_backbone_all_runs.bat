@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
if exist "%ALLRUN_BACKBONE_OUTPUT%\completed.json" (
  echo Completed all-runs backbone exists; skip without overwrite: %ALLRUN_BACKBONE_OUTPUT%
  exit /b 0
)
"%PYTHON_BIN%" tools\train_backbone.py ^
  --dataset-root "%DATASET_ROOT%" ^
  --protocol-root "%PROTOCOL_ROOT%" ^
  --train-scope all_runs ^
  --output-dir "%ALLRUN_BACKBONE_OUTPUT%" ^
  --camera-id "%CAMERA_ID%" ^
  --epochs %BACKBONE_EPOCHS% ^
  --batch-size 16 ^
  --num-workers %NUM_WORKERS% ^
  --seed %SEED% ^
  --amp
exit /b %ERRORLEVEL%
