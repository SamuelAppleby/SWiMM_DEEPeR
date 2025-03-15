@echo off
setlocal enabledelayedexpansion
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\..\.venv\Scripts\activate
cd ..\..\gym_underwater

set seeds=113 127 131 137 139
set algorithms=sac_noise_1
set render=human

for %%x in (%algorithms%) do (
    for /f "tokens=1,2,3 delims=_" %%a in ("%%x") do (
        set n=%%a
        set s=%%b
        set l=%%c
    )

    set w=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\models\!n!_!s!\!n!_!s!_!l!\best_model.zip

    for %%s in (%seeds%) do (
        for %%r in (%render%) do (
            python inference.py --seed %%s --algorithm !n! --render %%r --pre_trained_model_path !w! --compute_stats
@REM             python email_notifier.py --msg %%s
        )
    )
)

exit
