@echo off
set "FILE_PATH=train_output.txt"
call ..\.venv\Scripts\activate
python ..\gym_underwater\train.py
pause
del "%FILE_PATH%"
exit