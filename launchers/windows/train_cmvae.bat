@echo off
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\..\.venv\Scripts\activate
cd ..\..\cmvae_scripts
set files=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_47.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_53.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_59.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_61.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_67.yml

for %%i in (%files%) do (
    python ..\gym_underwater\change_config.py --source_file %%i --dest_file C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config.yml
    python train_cmvae.py
)

exit