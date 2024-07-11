export "PYTHONPATH=C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR;%PYTHONPATH%"
source ../../.venv/bin/activate
cd ../../gym_underwater

seeds=(97)

for seed in "${seeds[@]}"
do
    python inference.py --seed "$seed"
done

exit