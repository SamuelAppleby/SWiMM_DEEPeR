@echo off
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\..\.venv\Scripts\activate
cd ..\..\cmvae_scripts
python train_cmvae.py
exit