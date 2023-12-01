@echo off
call ..\.venv\Scripts\activate
cd ..\cmvae_scripts
start python train_cmvae.py --data_dir C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\vae_training_set --n_z 10 --epochs 200 --load_during_training
exit