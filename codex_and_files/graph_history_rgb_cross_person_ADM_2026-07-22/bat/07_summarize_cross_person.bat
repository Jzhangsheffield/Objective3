@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
"%PYTHON_BIN%" tools\summarize_cross_person.py ^
  --outputs-root "%OUTPUTS_ROOT%" ^
  --output-dir "%CROSS_PERSON_SUMMARY_ROOT%" ^
  --participants A D M
exit /b %ERRORLEVEL%

