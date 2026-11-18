@echo off
rem Opens the review window for the lab knife tiles Claude was unsure of (1480 tiles).
rem Each answer is saved to results\stage1_kd\lab_label\votes.csv right away; closing and reopening resumes at the first unanswered tile.
rem Keys: a/b/c = that box holds the knife, k = knife but no box fits, n = no knife, u = cannot tell,
rem       d then drag = draw your own knife box, z = undo, space/right = next, p/left = back, q = quit.
rem Add --all-flagged to also see every tile the votes disagreed on, including the ones Claude settled.
cd /d "%~dp0"
".venv\Scripts\python.exe" results\samples\poc\label_review_ui.py %*
pause
