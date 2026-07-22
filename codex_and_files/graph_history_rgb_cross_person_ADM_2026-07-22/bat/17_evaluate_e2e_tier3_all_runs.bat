@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
if exist "%ALLRUN_E2E_TIER3_OUTPUT%\completed.json" (
  echo Completed all-runs E2E Tier3 evaluation exists; skip without overwrite.
  exit /b 0
)
if not exist "%ALLRUN_BACKBONE_CKPT%" exit /b 1
"%PYTHON_BIN%" tools\evaluate_e2e_tier3.py ^
  --dataset-root "%DATASET_ROOT%" ^
  --protocol-root "%PROTOCOL_ROOT%" ^
  --checkpoint "%ALLRUN_BACKBONE_CKPT%" ^
  --output-dir "%ALLRUN_E2E_TIER3_OUTPUT%" ^
  --train-scope all_runs ^
  --camera-id "%CAMERA_ID%" ^
  --batch-size 16 ^
  --num-workers %NUM_WORKERS% ^
  --seed %SEED% ^
  --amp
exit /b %ERRORLEVEL%
