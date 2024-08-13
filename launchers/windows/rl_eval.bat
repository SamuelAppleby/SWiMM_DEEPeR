@echo off
setlocal enabledelayedexpansion
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\..\.venv\Scripts\activate
cd ..\..\gym_underwater

set seeds=97
set render=human
set dir=1

for %%d in (%dir%) do (
    set w=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\models\sac\sac_%%d\best_model.zip

    for %%s in (%seeds%) do (
        for %%r in (%render%) do (
            python inference.py --seed %%s --render %%r --pre_trained_model_path !w!
        )
    )
)

exit
