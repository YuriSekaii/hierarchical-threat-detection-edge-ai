@echo off
rem Opens the review window for the test knives Claude could not sort by visibility (clear / blurred / hidden).
rem Each answer is saved to results\stage1_kd\vis_review\human\votes.csv right away; reopening resumes at the first unanswered knife.
rem Keys: c = clear, you can tell it is a knife   b = blurred by motion   h = hidden or barely visible
rem       x = no knife   u = cannot tell   space/right = next, p/left = back, z = undo, q = quit.
cd /d "%~dp0"
".venv\Scripts\python.exe" results\samples\poc\label_review_ui.py --vis --dir results\stage1_kd\vis_review\human %*
pause
