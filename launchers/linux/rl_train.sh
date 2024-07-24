export "PYTHONPATH=C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR;%PYTHONPATH%"
source ../../.venv/bin/activate
cd ../../gym_underwater

seeds=(97)
algorithm="sac"
n_envs=1

for seed in "${seeds[@]}"
do
    python train.py --seed "$seed" --algorithm "$algorithm" --n_envs "$n_envs"
done

exit