@echo off
cd ..
start pipenv run python -m gym_underwater.train
cd builds\windows
SWiMM_DEEPeR.exe modeServerControl
exit