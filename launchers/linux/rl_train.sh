export "PYTHONPATH=C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR;%PYTHONPATH%"
source ../../.venv/bin/activate
cd ../../gym_underwater

seeds=(97)
algorithm="sac"

for seed in "${seeds[@]}"
do
    python train.py --seed "$seed" --algorithm "$algorithm"
done

exit