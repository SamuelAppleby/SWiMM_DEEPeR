SET PIPENV_VENV_IN_PROJECT=1
pipenv install 
CALL .\.venv\Scripts\activate.bat
START python train.py
START RLSimulation\Builds\Windows\ReinforcementLearningSimulation.exe server 127.0.0.1:60261

