export "PYTHONPATH=C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR;%PYTHONPATH%"
source ../../.venv/bin/activate
cd ../../gym_underwater
python train &
cd ../builds/linux
./SWiMM_DEEPeR.x86_64 modeServerControl
exit