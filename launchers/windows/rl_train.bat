@echo off
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\..\.venv\Scripts\activate
cd ..\..\gym_underwater

set seeds=97 101 103 107 109
set algorithm=td3
set n_envs=1
set render=human

for %%s in (%seeds%) do (
    python train.py --n_envs %n_envs% --seed %%s --algorithm %algorithm% --render %render%
    python email_notifier.py --msg %%s
)

exit