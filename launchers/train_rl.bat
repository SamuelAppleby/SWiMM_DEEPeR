@echo off
call ..\.venv\Scripts\activate
cd ..\gym_underwater
python train.py
cd ..\builds\windows
SWiMM_DEEPeR.exe modeServerControl
exit