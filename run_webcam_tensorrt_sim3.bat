@echo off
rem Webcam pipeline with the 3-model Stage 2b blend (sim3). The original models still run with run_webcam_tensorrt.bat.
call "%~dp0run_webcam_tensorrt.bat" --stage2-models sim3 %*
