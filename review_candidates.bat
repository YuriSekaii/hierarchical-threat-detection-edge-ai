@echo off
rem Opens the review window for the knife candidates Claude was unsure of (spots where a detector fires but the
rem lab label says "no knife", and boxes on the new empty-hand frames). Same keys as review_lab_labels.bat.
rem Each answer is saved to results\stage1_kd\cand_review\votes.csv right away; reopening resumes at the first unanswered tile.
rem Keys: a = the box holds the knife, k = knife but the box does not fit, n = no knife, u = cannot tell,
rem       d then drag = draw your own knife box, z = undo, space/right = next, p/left = back, q = quit.
cd /d "%~dp0"
".venv\Scripts\python.exe" results\samples\poc\label_review_ui.py --dir results\stage1_kd\cand_review %*
pause
