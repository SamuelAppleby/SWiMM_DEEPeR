CALL ../.venv/Scripts/activate
cd ../gym_underwater
START python inference.py
cd ..\Builds\Windows
START SWiMM_DEEPeR.exe server_control
EXIT