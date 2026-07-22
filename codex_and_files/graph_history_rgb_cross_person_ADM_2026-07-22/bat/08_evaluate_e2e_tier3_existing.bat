@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
if not exist "%BACKBONE_CKPT%" (
  echo Missing existing Tier-3 last.pth: %BACKBONE_CKPT%
  exit /b 1
)
if exist "%E2E_TIER3_OUTPUT%\completed.json" (
  echo Existing completed result found; skip without overwriting: %E2E_TIER3_OUTPUT%
  exit /b 0
)
"%PYTHON_BIN%" tools\evaluate_e2e_tier3.py ^
  --dataset-root "%DATASET_ROOT%" ^
  --protocol-root "%PROTOCOL_ROOT%" ^
  --checkpoint "%BACKBONE_CKPT%" ^
  --output-dir "%E2E_TIER3_OUTPUT%" ^
  --train-scope normal_only ^
  --camera-id "%CAMERA_ID%" ^
  --batch-size 16 ^
  --num-workers %NUM_WORKERS% ^
  --seed %SEED% ^
  --amp
exit /b %ERRORLEVEL%
