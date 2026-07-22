@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
if not exist "%BACKBONE_CKPT%" (
  echo Missing retrained backbone: %BACKBONE_CKPT%
  echo Run bat\02_train_backbone_normal_only.bat first.
  exit /b 1
)
if not exist "%FEATURE_ROOT%" mkdir "%FEATURE_ROOT%"

"%PYTHON_BIN%" tools\extract_features.py ^
  --dataset-root "%DATASET_ROOT%" ^
  --manifest "%PROTOCOL_ROOT%\all_runs\train.jsonl" ^
  --checkpoint "%BACKBONE_CKPT%" ^
  --output "%FEATURE_ROOT%\train_all.pt" ^
  --camera-id "%CAMERA_ID%" ^
  --batch-size 16 ^
  --num-workers %NUM_WORKERS% ^
  --seed %SEED% ^
  --amp
if errorlevel 1 exit /b 1

"%PYTHON_BIN%" tools\extract_features.py ^
  --dataset-root "%DATASET_ROOT%" ^
  --manifest "%PROTOCOL_ROOT%\normal_only\test_all.jsonl" ^
  --checkpoint "%BACKBONE_CKPT%" ^
  --output "%FEATURE_ROOT%\test_all.pt" ^
  --camera-id "%CAMERA_ID%" ^
  --batch-size 16 ^
  --num-workers %NUM_WORKERS% ^
  --seed %SEED% ^
  --amp
exit /b %ERRORLEVEL%

