@echo off
setlocal EnableExtensions
call "%~dp0\34_train_dynamic_epoch_shuffle_normal_only.bat" || exit /b 1
call "%~dp0\35_train_dynamic_epoch_shuffle_all_runs.bat" || exit /b 1
echo Completed dynamic epoch-shuffle models for %TEST_PARTICIPANT% seed=%SEED%.
exit /b 0
