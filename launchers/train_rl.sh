cd ..
pipenv run python -m gym_underwater.train "$@" &
cd ../builds/linux
./SWiMM_DEEPeR.x86_64 modeServerControl
exit