for seed in {0..9}
do
    echo ${dataset} ${seed}
    python chem_finetune.py --dataset=${1} --seed=${seed} --cl_exp_dir=transfer_exp/chem/chembl_filtered --cl_model_name=cl_model_10.pth --save=100ep --epoch=100
done