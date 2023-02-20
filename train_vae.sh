export PIPENV_VENV_IN_PROJECT=1
pipenv install 
source .venv/bin/activate
cd cmvae_scripts
python3 train_cmvae.py --data_dir /home/b7034806/Repositories/Codebases/SWiMM_DEEPeR/data/vae_training_set/64x64 --n_z 20 --epochs 50
exit