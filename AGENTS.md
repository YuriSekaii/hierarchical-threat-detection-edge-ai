# Project rules

## Reports

- When you test a new approach or run a benchmark, write the result back as a report in `reports/` before you finish, even when the result is negative.
- One report file covers one finding, approach or method. A completely new idea or a new kind of work gets a new file named `MODIFICATION_REPORT_<TOPIC>.md`; more results on the same approach go into its existing file.
- Do not split a finding out of a report it is already written in. Fewer files is better; a new file is only for a topic no existing report covers.
- Small things need no report: typos, small bug fixes, renames, formatting.
- A report states the question, what was run (script and command), the numbers in a table, what was not checked, and where the files are.

## Commits

- Commit and push every change, without asking. Leave out weights, datasets and other large generated files.

## Standing constraints

- Never overwrite or rename the lab recordings in `data/lab_recordings/` or the shipped weights in `weights/`. New weights get a new file name.
- Take 002 (Oct 31) is a test recording: do not train on it and call the result a clean test. `data/Ground_Truth_Dataset` is the Stage 1 test set: never train on it.
- TensorRT engines run at 384x640 on both PCs.
- Stage 2 tests use the three-model recipe (ST-GCN, PoseConv3D, distilled PoseConv3D), not a substitute model.
- `ultralytics` is only in `.venv`; on Windows train with `workers=0` unless the script has a `__main__` guard.
