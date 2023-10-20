cd ..
IF EXIST .venv RMDIR /S /Q .venv
mkdir .venv
pipenv install --dev