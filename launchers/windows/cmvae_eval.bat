@echo off
setlocal enabledelayedexpansion

set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\..\.venv\Scripts\activate
cd ..\..\cmvae_scripts

set dir=0 1
set seeds=29 31

for %%d in (%dir%) do (
    set w=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\models\cmvae\%%d\best_model\model.ckpt

    for %%s in (%seeds%) do (
        python eval_cmvae.py --seed %%s --weights_path !w!
    )
)

exit
