call ../.venv/Scripts/activate
cd ../gym_underwater
start python inference.py
cd ..\builds\windows
SWiMM_DEEPeR.exe modeServerControl
exit