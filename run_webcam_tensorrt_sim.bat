@echo off
rem Webcam pipeline with the Stage 2b models retrained on simulated skeletons (sim v2).
rem The original models still run with run_webcam_tensorrt.bat.
call "%~dp0run_webcam_tensorrt.bat" --stage2-models sim %*
