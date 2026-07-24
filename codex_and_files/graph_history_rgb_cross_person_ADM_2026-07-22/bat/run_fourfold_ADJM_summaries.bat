@echo off
setlocal
call "%~dp0\28_summarize_normal_only_ADJM_3seeds.bat" || exit /b 1
call "%~dp0\29_summarize_all_runs_ADJM_3seeds.bat" || exit /b 1
call "%~dp0\30_summarize_training_scope_comparison_ADJM_3seeds.bat" || exit /b 1
echo Completed A/D/J/M three-seed summaries.
exit /b 0
