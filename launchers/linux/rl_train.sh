export "PYTHONPATH=C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR;%PYTHONPATH%"
source ../../.venv/bin/activate
cd ../../gym_underwater

seeds=(97 101 103 107 109)
algorithm="sac"
n_envs=1
render="human"

for seed in "${seeds[@]}"
do
    python train.py --seed "$seed" --algorithm "$algorithm" --n_envs "$n_envs" --render "$render"
    python email_notifier.py --msg "$seed"
done

exit