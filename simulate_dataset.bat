@echo off
rem Skeleton simulator: generates simulated knife attacks, normal activity and perturbed copies
rem of the real training clips, in the training format, into data\Simulated\ (see metadata.csv
rem and previews\ there). Then checks them against the current Stage 2 cascade.
rem Fewer clips: simulate_dataset.bat --per-action 50 --normals 30 --real-aug 3
rem Another folder: simulate_dataset.bat --out data\Simulated_v2
setlocal
cd /d "%~dp0"
if exist ".\.venv\Scripts\python.exe" (
    set "PY_BIN=.\.venv\Scripts\python.exe"
) else (
    set "PY_BIN=python"
)
"%PY_BIN%" scripts\simulate_dataset.py %*
if %ERRORLEVEL% NEQ 0 (
    pause
    exit /b 1
)
set "SIM_OUT=data\Simulated"
:find_out
if "%~1"=="" goto run_check
if /I "%~1"=="--out" set "SIM_OUT=%~2"
shift
goto find_out
:run_check
"%PY_BIN%" scripts\validate_simulation.py --sim "%SIM_OUT%"
pause
endlocal
