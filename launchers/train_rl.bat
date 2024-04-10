@echo off
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
cd ..
start pipenv run python -m gym_underwater.train
cd builds\windows
SWiMM_DEEPeR.exe modeServerControl
exit