@echo off
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\..\.venv\Scripts\activate
cd ..\..\gym_underwater

set seeds=97
set algorithm=sac

for %%s in (%seeds%) do (
    python train.py --seed %%s --algorithm %algorithm%
)

exit