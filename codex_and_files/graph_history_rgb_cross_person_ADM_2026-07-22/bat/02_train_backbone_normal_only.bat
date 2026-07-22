@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
"%PYTHON_BIN%" tools\train_backbone.py ^
  --dataset-root "%DATASET_ROOT%" ^
  --protocol-root "%PROTOCOL_ROOT%" ^
  --train-scope normal_only ^
  --output-dir "%BACKBONE_OUTPUT%" ^
  --camera-id "%CAMERA_ID%" ^
  --epochs %BACKBONE_EPOCHS% ^
  --batch-size 16 ^
  --num-workers %NUM_WORKERS% ^
  --seed %SEED% ^
  --amp
exit /b %ERRORLEVEL%
