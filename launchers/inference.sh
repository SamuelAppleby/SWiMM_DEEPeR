source ../.venv/bin/activate
cd ../gym_underwater
python3 inference.py "$@" &
cd ../Builds/Linux
./SWiMM_DEEPeR.x86_64 mode_training
exit
