@echo off
cd /d "%~dp0"
REM ===========================================================================
REM  BOX (4) — Second GAM-RF union, all four arms at once.
REM  Input : the *_retainedvars_postcollin_*.csv files from box (3)
REM  Needs : jpn_pass2.txt  kor_pass2.txt  phl_pass2.txt  pooled_pass2.txt
REM ===========================================================================
echo BOX (4) Second feature selection - launching FOUR runs in parallel...
echo Cores available: %NUMBER_OF_PROCESSORS%   (each run gets about a quarter)
echo.
start "4 MASE"     run_arm.bat jpn_pass2.txt    gamrf_jpn_pass2.log    4
start "4 CHEORWON" run_arm.bat kor_pass2.txt    gamrf_kor_pass2.log    4
start "4 IRRI"     run_arm.bat phl_pass2.txt    gamrf_phl_pass2.log    4
start "4 POOLED"   run_arm.bat pooled_pass2.txt gamrf_pooled_pass2.log 4
echo.
echo Four windows opened. Do not close them.
timeout /t 12 >nul
