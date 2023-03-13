CALL .\..\.venv\Scripts\activate.bat
cd ..\cmvae_scripts
START python train_cmvae.py --data_dir D:\vae_training_set\64x64 --n_z 10 --epochs 50
EXIT