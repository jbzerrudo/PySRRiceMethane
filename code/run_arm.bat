@echo off
REM run_arm.bat <config.txt> <log> [concurrent runs]  -- run_one.bat, 4 edits only
setlocal
cd /d "%~dp0"
set "CFG=%~1"
set "LOG=%~2"
set "NRUNS=%~3"
if "%CFG%"=="" set "CFG=full_config.txt"
if "%LOG%"=="" set "LOG=gamrf_run.log"
if "%NRUNS%"=="" set "NRUNS=2"

REM --- write console/log as UTF-8 so the script's unicode chars don't crash ---
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM --- give this run its share of the logical processors (3rd arg, default 2) ---
set /a HALF=%NUMBER_OF_PROCESSORS%/%NRUNS%
if %HALF% LSS 1 set "HALF=1"
set "OMP_NUM_THREADS=%HALF%"
set "OPENBLAS_NUM_THREADS=%HALF%"
set "MKL_NUM_THREADS=%HALF%"
set "NUMEXPR_NUM_THREADS=%HALF%"
set "VECLIB_MAXIMUM_THREADS=%HALF%"
set "GAMRF_N_JOBS=%HALF%"

REM --- locate a Python interpreter ---
set "PYEXE="
for %%P in (python.exe) do set "PYEXE=%%~$PATH:P"
if not defined PYEXE if exist "%USERPROFILE%\anaconda3\python.exe"      set "PYEXE=%USERPROFILE%\anaconda3\python.exe"
if not defined PYEXE if exist "%USERPROFILE%\miniconda3\python.exe"     set "PYEXE=%USERPROFILE%\miniconda3\python.exe"
if not defined PYEXE if exist "%LOCALAPPDATA%\anaconda3\python.exe"     set "PYEXE=%LOCALAPPDATA%\anaconda3\python.exe"
if not defined PYEXE if exist "C:\ProgramData\Anaconda3\python.exe"     set "PYEXE=C:\ProgramData\Anaconda3\python.exe"
if not defined PYEXE ( where py >nul 2>&1 && set "PYEXE=py" )

if not defined PYEXE (
  echo.
  echo Could not find Python automatically.
  echo Open run_one.bat and set this line to your interpreter, e.g.:
  echo     set "PYEXE=C:\Users\zerru001\anaconda3\python.exe"
  echo.
  pause
  exit /b 1
)

REM --- verify the required packages live in this Python BEFORE the long run ---
"%PYEXE%" -c "import numpy,pandas,scipy,sklearn,matplotlib,pygam" 1>nul 2>nul
if errorlevel 1 (
  echo.
  echo This Python is missing required packages ^(e.g. pygam^):
  echo     %PYEXE%
  echo Install them into it:
  echo     "%PYEXE%" -m pip install pygam scikit-learn scipy pandas numpy matplotlib
  echo or edit run_one.bat and set PYEXE to the Python you normally use.
  echo.
  pause
  exit /b 1
)

echo ============================================================
echo  Python : %PYEXE%
echo  Cores  : %HALF% of %NUMBER_OF_PROCESSORS%   ^(%NRUNS% runs sharing^)
echo  Config : %CFG%
echo  Log    : %LOG%
echo ============================================================
echo Running. KEEP THIS WINDOW OPEN until it says Finished.
echo.
"%PYEXE%" 1_4GAM_RF_union.py "%CFG%" > "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (
  echo *** This run stopped with an error ^(exit %RC%^). Last 30 log lines: ***
  echo.
  powershell -NoProfile -Command "if (Test-Path '%LOG%') { Get-Content '%LOG%' -Tail 30 }"
) else (
  echo Finished OK.
)
echo.
pause
endlocal
