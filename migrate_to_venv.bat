@echo off
setlocal
cd /d "%~dp0"
title Migrate Packages to .venv (Zero Download)

echo ===============================================================================
echo   HIERARCHICAL EDGE-AI THREAT AND VIOLENCE DETECTION SYSTEM
echo   MIGRATION UTILITY: HOST PC -^> LOCAL .VENV (ZERO RE-DOWNLOAD)
echo ===============================================================================
echo.
echo [INFO] This utility will:
echo        1. Create an isolated project virtual environment (.venv) if missing
echo        2. Move all installed AI packages (PyTorch, TensorRT, YOLO, etc.) into .venv
echo        3. Remove those packages from your PC host storage (freeing C: drive space)
echo        4. Zero re-download needed (takes ~2 seconds)
echo.

python scripts\migrate_packages_to_venv.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] Host Python returned an error code during migration.
)

echo.
echo ===============================================================================
echo [DONE] You can now run:
echo        - run_webcam_tensorrt.bat  (NVIDIA RTX 3050 TensorRT GPU Mode)
echo        - run_webcam_pytorch.bat   (Universal PyTorch Eager / Laptop Mode)
echo ===============================================================================
echo.
pause
