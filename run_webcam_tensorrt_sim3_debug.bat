@echo off
rem Debug mode (results\debug_stage2\) with the 3-model Stage 2b blend (sim3).
call "%~dp0run_webcam_tensorrt.bat" --stage2-models sim3 --debug %*
