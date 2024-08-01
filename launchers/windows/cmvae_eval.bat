@echo off
setlocal enabledelayedexpansion
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\..\.venv\Scripts\activate
cd ..\..\cmvae_scripts

set dir=5 6 7 8 9
set seeds=71 73 79 83 89

for %%d in (%dir%) do (
    set w=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\models\cmvae\%%d\best_model\model.ckpt

    for %%s in (%seeds%) do (
        python eval_cmvae.py --seed %%s --weights_path !w!
    )
)

exit