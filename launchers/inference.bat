CALL ../.venv/Scripts/activate
cd ../gym_underwater
START python inference.py
cd ..\builds\windows
START SWiMM_DEEPeR.exe mode_server_control
EXIT