CALL ../.venv/Scripts/activate
cd ../gym_underwater
START python train.py
cd ..\SWiMM_DEEPeR\Builds\Windows
START SWiMM_DEEPeR.exe mode_training
EXIT