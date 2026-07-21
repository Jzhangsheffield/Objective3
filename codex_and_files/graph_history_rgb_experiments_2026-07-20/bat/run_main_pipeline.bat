@echo off
setlocal
call "%~dp0\01_prepare_protocols.bat" || exit /b 1
call "%~dp0\03_extract_features_existing_last.bat" || exit /b 1
call "%~dp0\04_train_main_m0_m6.bat" || exit /b 1
call "%~dp0\06_summarize_results.bat" || exit /b 1
echo Main normal-only J-as-test pipeline completed.
exit /b 0
