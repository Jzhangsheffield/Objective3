@echo off
setlocal EnableExtensions
call "%~dp0\31_train_direct_head_fusion_normal_only.bat" || exit /b 1
call "%~dp0\32_train_direct_head_fusion_all_runs.bat" || exit /b 1
echo Completed direct-head fusion models for %TEST_PARTICIPANT% seed=%SEED%.
exit /b 0
