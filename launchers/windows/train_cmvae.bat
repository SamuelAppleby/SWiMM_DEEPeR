@echo off
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\..\.venv\Scripts\activate
cd ..\..\cmvae_scripts
set files=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_11.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_13.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_17.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_19.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_23.yml

for %%i in (%files%) do (
    python ..\gym_underwater\change_config.py --source_file %%i --dest_file C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config.yml
    python train_cmvae.py
)

exit