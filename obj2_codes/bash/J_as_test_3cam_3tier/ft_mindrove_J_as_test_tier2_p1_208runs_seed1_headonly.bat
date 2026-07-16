@echo off
call "%~dp0batch_run_mindrove_J_as_test_all_tiers_lengths.bat" head_only tier2
exit /b %ERRORLEVEL%
