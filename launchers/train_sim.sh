source ../.venv/Scripts/activate
cd ../gym_underwater
python3 train.py "$@" &
cd ../SWiMM_DEEPeR/Builds/Linux
SWiMM_DEEPeR.x86_64 mode_training -batchmode
exit