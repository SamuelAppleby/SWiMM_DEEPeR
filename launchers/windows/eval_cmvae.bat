@echo off
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\..\.venv\Scripts\activate
cd ..\..\cmvae_scripts
set files=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\cmvae\cmvae_inference_config_5.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\cmvae\cmvae_inference_config_6.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\cmvae\cmvae_inference_config_7.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\cmvae\cmvae_inference_config_8.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\cmvae\cmvae_inference_config_9.yml
set seeds=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_71.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_73.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_79.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_83.yml C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config_89.yml

for %%i in (%files%) do (
    python ..\gym_underwater\change_config.py --source_file %%i --dest_file C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\cmvae\cmvae_inference_config.yml
    for %%j in (%seeds%) do (
        python ..\gym_underwater\change_config.py --source_file %%j --dest_file C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\configs\env_config.yml
        python eval_cmvae.py
    )
)

exit