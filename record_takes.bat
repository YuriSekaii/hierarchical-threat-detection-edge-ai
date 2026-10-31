@echo off
rem Lab take recorder: records the take plan for the skeleton simulator.
rem Close the detection pipeline first (the camera can only be used by one program).
rem Files go to data\lab_recordings\DATE\ (calib, check, test, normal) with takes.csv
rem Other camera: record_takes.bat --camera 1
setlocal
cd /d "%~dp0"
if exist ".\.venv\Scripts\python.exe" (
    set "PY_BIN=.\.venv\Scripts\python.exe"
) else (
    set "PY_BIN=python"
)
"%PY_BIN%" scripts\record_takes.py %*
if %ERRORLEVEL% NEQ 0 pause
endlocal
