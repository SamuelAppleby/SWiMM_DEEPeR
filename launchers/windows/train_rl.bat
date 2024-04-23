@echo off
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\..\.venv\Scripts\activate
cd ..\..\gym_underwater
start python train.py
cd ..\builds\windows
SWiMM_DEEPeR.exe modeServerControl debugLogs
exit