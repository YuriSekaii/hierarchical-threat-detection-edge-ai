@echo off
rem Same as run_webcam_tensorrt.bat, but with the Stage 2 debug tracer enabled.
rem Logs and analyzed frames go to results\debug_stage2\session_TIMESTAMP
call "%~dp0run_webcam_tensorrt.bat" --debug %*
