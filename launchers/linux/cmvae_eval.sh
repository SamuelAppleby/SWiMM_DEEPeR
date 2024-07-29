export "PYTHONPATH=C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR;%PYTHONPATH%"
source ../../.venv/bin/activate
cd ../../cmvae_scripts

dirs=(0 1 2 3 4)
seeds=(29 31 37 41 43)

for dir in "${dirs[@]}"
do
    w="<dir>"

    for seed in "${seeds[@]}"
    do
        python eval_cmvae.py --seed "$seed" --weights_path "$w"
    done
done

exit