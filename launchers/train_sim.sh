source ../.venv/bin/activate
cd ../gym_underwater
python3 train.py "$@" &
cd ../builds/linux
SWiMM_DEEPeR.x86_64 mode_server_control
exit
