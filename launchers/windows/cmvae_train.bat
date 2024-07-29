@echo off
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\..\.venv\Scripts\activate
cd ..\..\cmvae_scripts

set seeds=11 13 17 19 23

for %%s in (%seeds%) do (
    python train_cmvae.py --seed %%s
)

exit