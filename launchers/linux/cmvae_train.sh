export "PYTHONPATH=C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR;%PYTHONPATH%"
source ../../.venv/bin/activate
cd ../../cmvae_scripts

seeds=(11 13)

for seed in "${seeds[@]}"
do
    python train_cmvae.py --seed "$seed"
done

exit