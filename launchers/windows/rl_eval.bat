@echo off
setlocal enabledelayedexpansion
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\..\.venv\Scripts\activate
cd ..\..\gym_underwater

set seeds=113 127 131 137 139
set algorithm=td3
set render=human
set dir=4

for %%d in (%dir%) do (
    set w=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\models\%algorithm%\%algorithm%_%%d\best_model.zip

    for %%s in (%seeds%) do (
        for %%r in (%render%) do (
            python inference.py --seed %%s --algorithm %algorithm% --render %%r --pre_trained_model_path !w!
            python email_notifier.py --msg %%s
        )
    )
)

exit
