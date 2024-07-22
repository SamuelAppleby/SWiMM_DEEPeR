cd ../..
if [ -d ".venv" ]; then rm -Rf .venv; fi
mkdir .venv
pipenv install
pipenv run pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
