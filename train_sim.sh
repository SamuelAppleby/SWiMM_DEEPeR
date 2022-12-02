export PIPENV_VENV_IN_PROJECT=1
pipenv install 
source .venv/bin/activate
python3 train.py "$@" &
RLSimulation/Builds/Linux/ReinforcementLearningSimulation.x86_64 server 127.0.0.1:60261 debug_conf_dir Configs/data/client_debug_config_build_linux.json network_conf_dir Configs/data/network_config.json
exit