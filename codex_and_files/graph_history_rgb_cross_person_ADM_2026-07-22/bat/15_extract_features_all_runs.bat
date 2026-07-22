@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
if exist "%ALLRUN_FEATURE_ROOT%\completed.json" (
  echo Completed all-runs feature cache exists; skip without overwrite: %ALLRUN_FEATURE_ROOT%
  exit /b 0
)
if not exist "%ALLRUN_BACKBONE_CKPT%" (
  echo Missing all-runs backbone: %ALLRUN_BACKBONE_CKPT%
  exit /b 1
)
"%PYTHON_BIN%" tools\guard_output_dir.py --output-dir "%ALLRUN_FEATURE_ROOT%" || exit /b 1
"%PYTHON_BIN%" tools\extract_features.py ^
  --dataset-root "%DATASET_ROOT%" ^
  --manifest "%PROTOCOL_ROOT%\all_runs\train.jsonl" ^
  --checkpoint "%ALLRUN_BACKBONE_CKPT%" ^
  --output "%ALLRUN_FEATURE_ROOT%\train_all.pt" ^
  --camera-id "%CAMERA_ID%" ^
  --batch-size 16 ^
  --num-workers %NUM_WORKERS% ^
  --seed %SEED% ^
  --amp
if errorlevel 1 exit /b 1
"%PYTHON_BIN%" tools\extract_features.py ^
  --dataset-root "%DATASET_ROOT%" ^
  --manifest "%PROTOCOL_ROOT%\all_runs\test_all.jsonl" ^
  --checkpoint "%ALLRUN_BACKBONE_CKPT%" ^
  --output "%ALLRUN_FEATURE_ROOT%\test_all.pt" ^
  --completion-marker "%ALLRUN_FEATURE_ROOT%\completed.json" ^
  --camera-id "%CAMERA_ID%" ^
  --batch-size 16 ^
  --num-workers %NUM_WORKERS% ^
  --seed %SEED% ^
  --amp
exit /b %ERRORLEVEL%
