@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
if exist "%FEATURE_ROOT%\completed.json" (
  echo Completed normal-only feature cache exists; skip without overwrite: %FEATURE_ROOT%
  exit /b 0
)
if not exist "%BACKBONE_CKPT%" (
  echo Missing normal-only backbone: %BACKBONE_CKPT%
  exit /b 1
)
"%PYTHON_BIN%" tools\guard_output_dir.py --output-dir "%FEATURE_ROOT%" || exit /b 1
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
  --completion-marker "%FEATURE_ROOT%\completed.json" ^
  --camera-id "%CAMERA_ID%" ^
  --batch-size 16 ^
  --num-workers %NUM_WORKERS% ^
  --seed %SEED% ^
  --amp
exit /b %ERRORLEVEL%
