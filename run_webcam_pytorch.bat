@echo off
setlocal
cd /d "%~dp0"
title Hierarchical Threat Detection - Base PyTorch Eager (Webcam)

echo ===============================================================================
echo   HIERARCHICAL EDGE-AI THREAT AND VIOLENCE DETECTION SYSTEM
echo   MODE: BASE PYTORCH EAGER (.pt / .pth REFERENCE MODELS)
echo ===============================================================================
echo.
echo [INFO] Pipeline Stack:
echo        - Stage 1: Distilled YOLO26s (PyTorch .pt Checkpoint, Hungarian NMS-Free)
echo        - Stage 2a: YOLO26s-Pose + ByteTrack Actor Locking (PyTorch .pt Checkpoint)
echo        - Stage 2b: Champion 2 Staircase Cascade (PyTorch .pth Weights)
echo        - Buffer:  In-Memory TurboJPEG Ring Buffer (~188.5 MB RAM)
echo.
echo [INFO] Hardware Support: Universal (NVIDIA GPU if available, CPU if on laptop)
echo [INFO] Ingesting real-time video from Webcam (Index: 0)
echo [INFO] Press 'q' in the camera window to safely terminate surveillance.
echo.

if exist ".\.venv\Scripts\python.exe" (
    set "PY_BIN=.\.venv\Scripts\python.exe"
) else if exist "..\.venv\Scripts\python.exe" (
    set "PY_BIN=..\.venv\Scripts\python.exe"
) else (
    set "PY_BIN=python"
)

echo [CHECK] Verifying Python environment and dependencies...

"%PY_BIN%" -c "import torch, torchvision, ultralytics, cv2, scipy, sklearn" 2>nul
if %ERRORLEVEL% EQU 0 goto :LAUNCH

echo.
echo ===============================================================================
echo [AUTO-SETUP] Required libraries (PyTorch, Ultralytics, OpenCV, etc.) not found!
echo [AUTO-SETUP] Automatically installing all dependencies via pip...
echo              This is a one-time automated setup. Please wait...
echo ===============================================================================
echo.
"%PY_BIN%" -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [RETRY] Direct installation of core packages...
    "%PY_BIN%" -m pip install torch torchvision ultralytics opencv-python numpy scipy scikit-learn matplotlib pyyaml tqdm timm einops
)
if %ERRORLEVEL% NEQ 0 goto :INSTALL_DEPS_FAIL

echo.
echo [AUTO-SETUP] All dependencies installed successfully!
echo.

:LAUNCH
echo [LAUNCH] Starting PyTorch Baseline Surveillance Pipeline...
echo.
"%PY_BIN%" src/inference_production_pipeline.py --source 0 --conf 0.45 --backend pytorch %*
if %ERRORLEVEL% NEQ 0 goto :RUN_FAIL
goto :END

:INSTALL_DEPS_FAIL
echo.
echo ===============================================================================
echo [ERROR] Failed to automatically install dependencies.
echo Please ensure your internet connection is active and run manually:
echo   "%PY_BIN%" -m pip install -r requirements.txt
echo ===============================================================================
pause
exit /b 1

:RUN_FAIL
echo.
echo ===============================================================================
echo [ERROR] Pipeline stopped with exit code %ERRORLEVEL%.
echo Please inspect the error traceback above.
echo ===============================================================================
pause
exit /b %ERRORLEVEL%

:END
echo.
echo [INFO] Surveillance session terminated.
pause
endlocal
