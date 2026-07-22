@echo off
setlocal
call "%~dp0config_windows.bat"
echo ============================================================
echo Cross-person fold: %TEST_PARTICIPANT%-as-test, seed=%SEED%
echo ============================================================
call "%~dp0\00_validate_setup.bat" || exit /b 1
call "%~dp0\01_prepare_protocols.bat" || exit /b 1
call "%~dp0\02_train_backbone_normal_only.bat" || exit /b 1
call "%~dp0\03_extract_features_retrained_last.bat" || exit /b 1
call "%~dp0\04_train_main_m0_m6.bat" || exit /b 1
if "%RUN_AUXILIARY%"=="1" (
  call "%~dp0\05_train_aux_all_runs_m0_m6.bat"
  if errorlevel 1 exit /b 1
)
call "%~dp0\06_summarize_results.bat" || exit /b 1
echo Completed %TEST_PARTICIPANT%-as-test.
exit /b 0
