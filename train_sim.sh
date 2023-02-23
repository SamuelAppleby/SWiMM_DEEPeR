export PIPENV_VENV_IN_PROJECT=1
pipenv install 
source .venv/bin/activate
cd gym_underwater
python3 train.py "$@" &
cd ../SWiMM_DEEPeR/Builds/Linux
./SWiMM_DEEPeR.x86_64 training 
exit
