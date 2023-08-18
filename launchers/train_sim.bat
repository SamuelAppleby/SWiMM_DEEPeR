CALL ../.venv/Scripts/activate
cd ../gym_underwater
START python train.py
cd ..\builds\windows
START SWiMM_DEEPeR.exe mode_server_control
EXIT