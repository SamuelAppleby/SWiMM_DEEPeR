export PIPENV_VENV_IN_PROJECT=1
pipenv install 
source .venv/bin/activate
cd cmvae_scripts
python3 train_cmvae.py "$@" &
exit