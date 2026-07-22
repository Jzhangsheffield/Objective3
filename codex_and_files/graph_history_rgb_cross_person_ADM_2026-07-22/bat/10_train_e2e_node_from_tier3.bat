@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
if not exist "%BACKBONE_CKPT%" (
  echo Missing existing Tier-3 last.pth: %BACKBONE_CKPT%
  exit /b 1
)
if exist "%E2E_NODE_TRANSFER_OUTPUT%\completed.json" (
  echo Existing completed result found; skip without overwriting: %E2E_NODE_TRANSFER_OUTPUT%
  exit /b 0
)
"%PYTHON_BIN%" tools\train_e2e_node.py ^
  --dataset-root "%DATASET_ROOT%" ^
  --protocol-root "%PROTOCOL_ROOT%" ^
  --task-graph "%TASK_GRAPH%" ^
  --relation-matrix "%RELATION_MATRIX%" ^
  --output-dir "%E2E_NODE_TRANSFER_OUTPUT%" ^
  --init tier3 ^
  --init-checkpoint "%BACKBONE_CKPT%" ^
  --train-scope normal_only ^
  --camera-id "%CAMERA_ID%" ^
  --epochs %E2E_NODE_EPOCHS% ^
  --batch-size 16 ^
  --num-workers %NUM_WORKERS% ^
  --learning-rate %E2E_NODE_LR% ^
  --seed %SEED% ^
  --amp
exit /b %ERRORLEVEL%
