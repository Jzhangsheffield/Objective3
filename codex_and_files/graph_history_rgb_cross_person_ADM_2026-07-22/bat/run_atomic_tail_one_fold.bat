@echo off
setlocal EnableExtensions
call "%~dp0\37_train_atomic_tail_normal_only.bat" || exit /b 1
call "%~dp0\38_train_atomic_tail_all_runs.bat" || exit /b 1
echo Completed atomic-tail models for %TEST_PARTICIPANT% seed=%SEED%.
exit /b 0
