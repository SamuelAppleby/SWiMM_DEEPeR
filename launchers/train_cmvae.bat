CALL .\..\.venv\Scripts\activate.bat
cd ..\cmvae_scripts
START python train_cmvae.py --data_dir C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\vae_training_set\64x64 --n_z 10 --epochs 50
EXIT