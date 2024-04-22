cd ../..
if [ -d ".venv" ]; then rm -Rf .venv; fi
mkdir .venv
pipenv install