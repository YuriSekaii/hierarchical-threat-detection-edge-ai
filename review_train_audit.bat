@echo off
rem Opens the review window for the Stage 1 training label audit items (wrong_pos, missed_knife, bad_box, unclear).
rem Answers save immediately to results\stage1_kd\train_audit\human_review\votes.csv.
rem Keys: a/b/c = box holds knife, d then drag = draw your own box, n = no knife, u = cannot tell, z = undo, space/right = next, p/left = back, q = quit.
cd /d "%~dp0"
".venv\Scripts\python.exe" results\samples\poc\label_review_ui.py --dir results\stage1_kd\train_audit\human_review %*
pause
