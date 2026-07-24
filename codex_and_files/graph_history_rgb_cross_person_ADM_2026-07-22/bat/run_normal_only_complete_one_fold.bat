@echo off
setlocal
call "%~dp0config_windows.bat"
echo ============================================================
echo Strict normal-only pipeline: %TEST_PARTICIPANT%-as-test, seed=%SEED%
echo Existing completed outputs are skipped; non-empty incomplete outputs stop.
echo ============================================================
call "%~dp0\00_validate_setup.bat" || exit /b 1
call "%~dp0\13_prepare_protocols_all_runs_safe.bat" || exit /b 1
call "%~dp0\24_train_backbone_normal_only_safe.bat" || exit /b 1
call "%~dp0\25_extract_features_normal_only_safe.bat" || exit /b 1
call "%~dp0\26_train_normal_only_m0_m6_safe.bat" || exit /b 1
call "%~dp0\08_evaluate_e2e_tier3_existing.bat" || exit /b 1
call "%~dp0\09_train_e2e_node_scratch.bat" || exit /b 1
call "%~dp0\10_train_e2e_node_from_tier3.bat" || exit /b 1
call "%~dp0\27_summarize_normal_only_fold.bat" || exit /b 1
if "%RUN_SCOPE_COMPARISON%"=="1" (
  call "%~dp0\21_summarize_training_scope_comparison_fold.bat" || exit /b 1
)
echo Completed strict normal-only pipeline for %TEST_PARTICIPANT%-as-test seed=%SEED%.
exit /b 0
