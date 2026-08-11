@echo off
cd /d "%~dp0"
REM ===========================================================================
REM  BOX (1) — First GAM-RF union, all four arms at once.
REM  Input : the *_box0.csv files from 0_prep_arm_inputs.py
REM  Needs : jpn_pass1.txt  kor_pass1.txt  phl_pass1.txt  pooled_pass1.txt
REM ===========================================================================
echo BOX (1) First feature selection - launching FOUR runs in parallel...
echo Cores available: %NUMBER_OF_PROCESSORS%   (each run gets about a quarter)
echo.
start "1 MASE"     run_arm.bat jpn_pass1.txt    gamrf_jpn_pass1.log    4
start "1 CHEORWON" run_arm.bat kor_pass1.txt    gamrf_kor_pass1.log    4
start "1 IRRI"     run_arm.bat phl_pass1.txt    gamrf_phl_pass1.log    4
start "1 POOLED"   run_arm.bat pooled_pass1.txt gamrf_pooled_pass1.log 4
echo.
echo Four windows opened. Do not close them.
timeout /t 12 >nul
