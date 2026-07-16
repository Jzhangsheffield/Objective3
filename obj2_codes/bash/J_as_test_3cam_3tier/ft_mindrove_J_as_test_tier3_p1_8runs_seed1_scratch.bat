@echo off
call "%~dp0batch_run_mindrove_J_as_test_all_tiers_lengths.bat" scratch tier3
exit /b %ERRORLEVEL%
