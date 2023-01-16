SET PIPENV_VENV_IN_PROJECT=1
pipenv install 
CALL .\.venv\Scripts\activate.bat
cd cmvae_scripts
START python train_cmvae.py
EXIT