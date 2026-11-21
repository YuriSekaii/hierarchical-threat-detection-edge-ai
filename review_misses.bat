@echo off
rem Opens the review window for clear test knives the models miss, where the voters and Claude could not settle the label,
rem or where the new result differs from your earlier answer in review_visibility.bat (shown in orange on the right).
rem Each answer is saved to results\stage1_kd\vis_review\miss_check\human\votes.csv right away; reopening resumes.
rem Keys: c = clear knife in the green box   b = blurred by motion   h = hidden or barely visible
rem       x = no knife   w = knife clearly there, but the green box misses it   u = cannot tell
rem       space/right = next, p/left = back, z = undo, q = quit.
cd /d "%~dp0"
".venv\Scripts\python.exe" results\samples\poc\label_review_ui.py --vis --dir results\stage1_kd\vis_review\miss_check\human %*
pause
