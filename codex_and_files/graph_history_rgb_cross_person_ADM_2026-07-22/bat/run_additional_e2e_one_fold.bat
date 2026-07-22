@echo off
setlocal
call "%~dp0config_windows.bat"
echo ============================================================
echo Incremental E2E baselines: %TEST_PARTICIPANT%-as-test, seed=%SEED%
echo Existing backbone/features/M0-M6 will not be trained or overwritten.
echo ============================================================
call "%~dp0\00_validate_setup.bat" || exit /b 1
if not exist "%BACKBONE_CKPT%" (
  echo Missing existing Tier-3 last.pth: %BACKBONE_CKPT%
  exit /b 1
)
if not exist "%PROTOCOL_ROOT%\normal_only\train.jsonl" (
  echo Missing existing protocol: %PROTOCOL_ROOT%\normal_only\train.jsonl
  exit /b 1
)
for %%M in (m0 m1 m2 m3 m4 m5 m6) do (
  if not exist "%MODEL_ROOT%\normal_only\%%M\last.pth" (
    echo Missing existing %%M checkpoint: %MODEL_ROOT%\normal_only\%%M\last.pth
    exit /b 1
  )
)
call "%~dp0\08_evaluate_e2e_tier3_existing.bat" || exit /b 1
call "%~dp0\09_train_e2e_node_scratch.bat" || exit /b 1
call "%~dp0\10_train_e2e_node_from_tier3.bat" || exit /b 1
call "%~dp0\11_summarize_all_models_fold.bat" || exit /b 1
echo Completed incremental baselines for %TEST_PARTICIPANT%-as-test.
exit /b 0
