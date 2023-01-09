export PIPENV_VENV_IN_PROJECT=1
pipenv install 
source .venv/bin/activate
python3 train.py "$@" &
cd SWiMM_DEEPeR/Builds/Linux
SWiMM_DEEPeR.x86_64 debug_conf_dir training
exit