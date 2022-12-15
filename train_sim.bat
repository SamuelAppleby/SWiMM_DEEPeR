SET PIPENV_VENV_IN_PROJECT=1
pipenv install 
CALL .\.venv\Scripts\activate.bat
START python train.py
cd RLSimulation\Builds\Windows
START ReinforcementLearningSimulation.exe automation_training
EXIT