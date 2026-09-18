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

if exist ".\.venv\Scripts\python.exe" (
    set "PY_BIN=.\.venv\Scripts\python.exe"
) else if exist "..\.venv\Scripts\python.exe" (
    set "PY_BIN=..\.venv\Scripts\python.exe"
) else (
    set "PY_BIN=python"
)

echo [LAUNCH] Starting TensorRT Surveillance Pipeline...
echo.
"%PY_BIN%" src/inference_production_pipeline.py --source 0 --conf 0.45 --backend tensorrt %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ===============================================================================
    echo [ERROR] Pipeline stopped with exit code %ERRORLEVEL%.
    echo Please inspect the error traceback above.
    echo ===============================================================================
    pause
)

endlocal
