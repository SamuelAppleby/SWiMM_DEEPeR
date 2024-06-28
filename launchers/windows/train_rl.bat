@echo off
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\..\.venv\Scripts\activate
cd ..\..\gym_underwater
set files=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_97.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_101.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_103.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_107.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_109.yml

for %%i in (%files%) do (
    python change_config.py --source_file %%i --dest_file C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config.yml
    python train.py
)

exit
