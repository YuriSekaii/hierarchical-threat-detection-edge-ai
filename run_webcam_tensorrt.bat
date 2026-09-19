@echo off
setlocal
cd /d "%~dp0"
title Hierarchical Threat Detection - Native TensorRT Accelerated (Webcam)

echo ===============================================================================
echo   HIERARCHICAL EDGE-AI THREAT AND VIOLENCE DETECTION SYSTEM
echo   MODE: NATIVE NVIDIA TENSORRT 11.3 (HIGH-PERFORMANCE INFERENCE)
echo ===============================================================================
echo.
echo [INFO] Pipeline Stack:
echo        - Stage 1: Distilled YOLO26s (TensorRT FP16/FP32 Engine, Hungarian NMS-Free)
echo        - Stage 2a: YOLO26s-Pose + ByteTrack Actor Locking (TensorRT Engine)
echo        - Stage 2b: Champion 2 Staircase Cascade (PoseConv3D T3 -^> DT3 -^> ST-GCN)
echo        - Buffer:  In-Memory TurboJPEG Ring Buffer (~188.5 MB RAM)
echo.
echo [INFO] Hardware Enforcement: Strict NVIDIA CUDA GPU Active (No CPU Fallback)
echo [INFO] Ingesting real-time video from Webcam (Index: 0)
echo [INFO] Press 'q' in the camera window to safely terminate surveillance.
echo.

rem Auto-create isolated .venv if not present
if not exist ".\.venv\Scripts\python.exe" (
    echo [AUTO-SETUP] No virtual environment found.
    echo [AUTO-SETUP] Creating isolated project environment in .venv...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Failed to create .venv. Falling back to system Python.
    )
)

if exist ".\.venv\Scripts\python.exe" (
    set "PY_BIN=.\.venv\Scripts\python.exe"
) else if exist "..\.venv\Scripts\python.exe" (
    set "PY_BIN=..\.venv\Scripts\python.exe"
) else (
    set "PY_BIN=python"
)

rem Auto-heal Pillow C-extension if corrupted
"%PY_BIN%" -c "from PIL import Image" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [AUTO-REPAIR] Corrupted Pillow (PIL) C-extension detected!
    echo [AUTO-REPAIR] Reinstalling clean Pillow binary wheel...
    "%PY_BIN%" -m pip install --force-reinstall --no-cache-dir pillow
)

rem 1. Check if core dependencies are installed
"%PY_BIN%" -c "import torch, torchvision, ultralytics, cv2, scipy, sklearn" 2>nul
if %ERRORLEVEL% EQU 0 goto :CHECK_NVIDIA

rem 1a. Try migrating existing packages from host PC into .venv (Zero Download!)
if exist "scripts\migrate_packages_to_venv.py" (
    echo [CHECK] Libraries not yet in .venv. Scanning host PC for existing packages to migrate...
    python scripts\migrate_packages_to_venv.py
    "%PY_BIN%" -c "import torch, torchvision, ultralytics, cv2, scipy, sklearn" 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo [SUCCESS] Packages migrated into .venv successfully! Zero network download used.
        goto :CHECK_NVIDIA
    )
)

echo.
echo ===============================================================================
echo [AUTO-SETUP] Core libraries (PyTorch, Ultralytics, OpenCV) not found in environment!
echo [AUTO-SETUP] Automatically installing dependencies via pip...
echo              (Uses local pip cache if previously downloaded. Please wait...)
echo ===============================================================================
echo.
"%PY_BIN%" -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [RETRY] Direct installation of core packages...
    "%PY_BIN%" -m pip install torch torchvision ultralytics opencv-python numpy scipy scikit-learn matplotlib pyyaml tqdm timm einops
)
if %ERRORLEVEL% NEQ 0 goto :INSTALL_DEPS_FAIL

echo.
echo [AUTO-SETUP] Core dependencies installed successfully!
echo.

:CHECK_NVIDIA
rem 2. Check if an NVIDIA GPU is physically present
nvidia-smi >nul 2>&1
if %ERRORLEVEL% NEQ 0 goto :NO_NVIDIA_GPU

rem 3. Check if TensorRT is installed
"%PY_BIN%" -c "import tensorrt" 2>nul
if %ERRORLEVEL% EQU 0 goto :CHECK_CUDA

echo.
echo ===============================================================================
echo [AUTO-SETUP] 'tensorrt' is not installed in the current Python environment.
echo [AUTO-SETUP] Automatically installing TensorRT 11.3: tensorrt-cu12, onnx, onnxslim
echo              This is a one-time automated setup for native GPU acceleration.
echo ===============================================================================
echo.
"%PY_BIN%" -m pip install tensorrt-cu12 onnx onnxslim
if %ERRORLEVEL% NEQ 0 goto :INSTALL_FAIL
echo.
echo [AUTO-SETUP] TensorRT dependencies installed successfully!
echo.

:CHECK_CUDA
rem 4. Check if PyTorch has CUDA enabled
"%PY_BIN%" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
if %ERRORLEVEL% EQU 0 goto :LAUNCH

echo.
echo ===============================================================================
echo [AUTO-SETUP] Detected CPU-only PyTorch build on this system!
echo [AUTO-SETUP] Your NVIDIA GPU requires PyTorch compiled with CUDA support.
echo [AUTO-SETUP] Automatically installing PyTorch with CUDA 12.4 for your GPU...
echo              (Uses local pip wheel cache if available. Please wait...)
echo ===============================================================================
echo.
"%PY_BIN%" -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124 --force-reinstall
if %ERRORLEVEL% NEQ 0 goto :CUDA_INSTALL_FAIL
echo.
echo [AUTO-SETUP] PyTorch CUDA installed successfully!
echo.

:LAUNCH
echo [LAUNCH] Starting TensorRT Surveillance Pipeline...
echo [NOTE]   Any missing .engine models will auto-compile from .onnx on first run.
echo.
"%PY_BIN%" src/inference_production_pipeline.py --source 0 --conf 0.45 --backend tensorrt %*
if %ERRORLEVEL% NEQ 0 goto :RUN_FAIL
goto :END

:NO_NVIDIA_GPU
echo.
echo ===============================================================================
echo [HARDWARE ALERT] No NVIDIA CUDA GPU detected on this computer!
echo Task Manager shows this machine only has integrated Intel Iris Xe / AMD graphics.
echo.
echo TensorRT and this surveillance pipeline strictly require an NVIDIA GPU
echo (GeForce RTX / GTX, Quadro, or Jetson).
echo.
echo To run on this laptop, please run:
echo   run_webcam_pytorch.bat
echo (Which runs in CPU reference mode without needing an NVIDIA card).
echo ===============================================================================
pause
exit /b 1

:INSTALL_DEPS_FAIL
echo.
echo ===============================================================================
echo [ERROR] Failed to automatically install dependencies.
echo Please ensure your internet connection is active and run manually:
echo   "%PY_BIN%" -m pip install -r requirements.txt
echo ===============================================================================
pause
exit /b 1

:INSTALL_FAIL
echo.
echo ===============================================================================
echo [ERROR] Automatic installation of tensorrt-cu12 failed.
echo Please ensure you have an active internet connection and run manually:
echo   "%PY_BIN%" -m pip install tensorrt-cu12 onnx onnxslim
echo ===============================================================================
pause
exit /b 1

:CUDA_INSTALL_FAIL
echo.
echo ===============================================================================
echo [ERROR] Failed to automatically install CUDA-enabled PyTorch.
echo Please run manually in your command prompt:
echo   "%PY_BIN%" -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124 --force-reinstall
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
