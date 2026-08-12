@echo off
cd /d "%~dp0"
echo Launching FULL and GROWING GAM-RF runs in parallel...
echo.
start "GAMRF FULL season"    run_one.bat full_config.txt gamrf_full.log
start "GAMRF GROWING season" run_one.bat gs_config.txt   gamrf_growingseason.log
echo.
echo Two windows opened (FULL and GROWING). Each prints its Python and core
echo count, then runs and STAYS OPEN. Do not close those two windows.
echo If a window shows a Python error, copy that text to Claude.
echo Logs are written here: gamrf_full.log and gamrf_growingseason.log
timeout /t 12 >nul
