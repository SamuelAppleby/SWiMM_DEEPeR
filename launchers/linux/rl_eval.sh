export "PYTHONPATH=C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR;%PYTHONPATH%"
source ../../.venv/bin/activate
cd ../../gym_underwater

seeds=(97)
render="human"

for seed in "${seeds[@]}"
do
    python inference.py --seed "$seed" --render "$render"
done

exit