SET PIPENV_VENV_IN_PROJECT=1
pipenv install 
CALL .\.venv\Scripts\activate.bat
START python train.py
START RLSimulation\Builds\Windows\ReinforcementLearningSimulation.exe debug_conf_dir Configs\data\client_debug_config_build_windows.json network_conf_dir Configs\data\network_config.json automation_training

