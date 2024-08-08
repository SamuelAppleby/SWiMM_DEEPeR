@echo off
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\..\.venv\Scripts\activate
cd ..\..\gym_underwater

set seeds=97

for %%s in (%seeds%) do (
    python inference.py --seed %%s --render none
)

exit