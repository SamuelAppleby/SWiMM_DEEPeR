SET PIPENV_VENV_IN_PROJECT=1
pipenv install 
CALL .\.venv\Scripts\activate.bat
START python train.py
cd SWiMM_DEEPeR\Builds\Windows
START SWiMM_DEEPeR.exe training
EXIT