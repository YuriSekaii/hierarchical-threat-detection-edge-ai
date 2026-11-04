@echo off
rem Debug mode (results\debug_stage2\) with the Stage 2b models retrained on simulated skeletons.
call "%~dp0run_webcam_tensorrt.bat" --stage2-models sim --debug %*
