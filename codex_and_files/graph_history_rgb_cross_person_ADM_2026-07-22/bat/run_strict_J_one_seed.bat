@echo off
setlocal
set "TEST_PARTICIPANT=J"
call "%~dp0config_windows.bat"
echo ============================================================
echo Strict J LOSO, seed=%SEED%: normal-only then all-runs
echo ============================================================
set "RUN_SCOPE_COMPARISON=0"
call "%~dp0run_normal_only_complete_one_fold.bat" || exit /b 1
set "RUN_SCOPE_COMPARISON=1"
call "%~dp0run_all_runs_one_fold.bat" || exit /b 1
echo Completed strict J LOSO seed=%SEED%.
exit /b 0
