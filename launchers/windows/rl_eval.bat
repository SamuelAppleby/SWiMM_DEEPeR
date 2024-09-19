@echo off
setlocal enabledelayedexpansion
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\..\.venv\Scripts\activate
cd ..\..\gym_underwater

set seeds=149 151 157 163 167
set algorithms=sac_1 ppo_3 td3_4
set render=human

for %%x in (%algorithms%) do (
    for /f "tokens=1,2 delims=_" %%a in ("%%x") do (
        set n=%%a
        set c=%%b
    )

    set w=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\models\!n!\!n!_!c!\best_model.zip

    for %%s in (%seeds%) do (
        for %%r in (%render%) do (
            python inference.py --seed %%s --algorithm !n! --render %%r --pre_trained_model_path !w! --compute_stats
            python email_notifier.py --msg %%s
        )
    )
)

exit
