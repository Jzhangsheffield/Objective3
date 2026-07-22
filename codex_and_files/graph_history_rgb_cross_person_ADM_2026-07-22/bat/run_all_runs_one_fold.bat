@echo off
setlocal
call "%~dp0config_windows.bat"
echo ============================================================
echo Complete all-runs pipeline: %TEST_PARTICIPANT%-as-test, seed=%SEED%
echo All normal-only outputs remain untouched.
echo ============================================================
call "%~dp0\00_validate_setup.bat" || exit /b 1
call "%~dp0\13_prepare_protocols_all_runs_safe.bat" || exit /b 1
call "%~dp0\14_train_backbone_all_runs.bat" || exit /b 1
call "%~dp0\15_extract_features_all_runs.bat" || exit /b 1
call "%~dp0\16_train_all_runs_m0_m6.bat" || exit /b 1
call "%~dp0\17_evaluate_e2e_tier3_all_runs.bat" || exit /b 1
call "%~dp0\18_train_e2e_node_scratch_all_runs.bat" || exit /b 1
call "%~dp0\19_train_e2e_node_from_tier3_all_runs.bat" || exit /b 1
call "%~dp0\20_summarize_all_runs_fold.bat" || exit /b 1
call "%~dp0\21_summarize_training_scope_comparison_fold.bat" || exit /b 1
echo Completed complete all-runs pipeline for %TEST_PARTICIPANT%-as-test.
exit /b 0
