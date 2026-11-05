@echo off
rem Records the sample movements for the motion generator. Close the detection pipeline first.
rem Files go to data\lab_recordings\DATE\ (train_attack, train_normal, test_attack, test_normal) with takes.csv
call "%~dp0record_takes.bat" --plan samples %*
