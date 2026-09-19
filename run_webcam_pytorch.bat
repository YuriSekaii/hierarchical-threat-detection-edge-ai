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

rem Auto-create isolated .venv if not present
if exist ".\.venv\Scripts\python.exe" goto :SELECT_VENV
echo [AUTO-SETUP] No virtual environment found.
echo [AUTO-SETUP] Creating isolated project environment in .venv...
python -m venv .venv
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Failed to create .venv. Falling back to system Python.
)

:SELECT_VENV
if exist ".\.venv\Scripts\python.exe" (
    set "PY_BIN=.\.venv\Scripts\python.exe"
) else if exist "..\.venv\Scripts\python.exe" (
    set "PY_BIN=..\.venv\Scripts\python.exe"
) else (
    set "PY_BIN=python"
)

echo [CHECK] Verifying Python environment and dependencies (%PY_BIN%)...

rem Pre-flight: Auto-heal Pillow C-extension if corrupted
"%PY_BIN%" -c "from PIL import Image" >nul 2>&1
if %ERRORLEVEL% EQU 0 goto :CHECK_CORE_DEPS

echo [AUTO-REPAIR] Corrupted Pillow C-extension detected.
echo [AUTO-REPAIR] Reinstalling clean Pillow binary wheel...
"%PY_BIN%" -m pip install --force-reinstall --no-cache-dir pillow

:CHECK_CORE_DEPS
"%PY_BIN%" -c "import torch, torchvision, ultralytics, cv2, scipy, sklearn" >nul 2>&1
if %ERRORLEVEL% EQU 0 goto :LAUNCH

rem Try migrating existing packages from host PC into .venv (Zero Download!)
if not exist "scripts\migrate_packages_to_venv.py" goto :RUN_DIAGNOSTICS
echo [CHECK] Core libraries not yet active in .venv. Scanning host PC for existing packages to migrate...
python scripts\migrate_packages_to_venv.py
"%PY_BIN%" -c "import torch, torchvision, ultralytics, cv2, scipy, sklearn" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Packages active in .venv! Zero network download used.
    goto :LAUNCH
)

:RUN_DIAGNOSTICS
echo.
echo [DIAGNOSTIC] Checking exact import status in environment:
"%PY_BIN%" -c "import torch; print('  [OK] torch ' + torch.__version__); import torchvision; print('  [OK] torchvision ' + torchvision.__version__); import ultralytics; print('  [OK] ultralytics ' + ultralytics.__version__); import cv2; print('  [OK] opencv ' + cv2.__version__); import scipy; print('  [OK] scipy ' + scipy.__version__); import sklearn; print('  [OK] scikit-learn ' + sklearn.__version__)"
echo.

echo ===============================================================================
echo [AUTO-SETUP] Missing core libraries detected in environment!
echo [AUTO-SETUP] Installing lightweight universal vision dependencies via pip...
echo              (NO TensorRT, pure CPU/Universal PyTorch stack)
echo ===============================================================================
echo.
"%PY_BIN%" -m pip install -r requirements.txt
if %ERRORLEVEL% EQU 0 goto :DEPS_OK

echo [RETRY] Direct installation of core packages...
"%PY_BIN%" -m pip install torch torchvision ultralytics opencv-python numpy scipy scikit-learn matplotlib seaborn pandas pyyaml tqdm timm einops pillow
if %ERRORLEVEL% NEQ 0 goto :INSTALL_DEPS_FAIL

:DEPS_OK
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
