# # pre-train
# python chem_pretrain.py --dataset_root=chem_data --dataset=chembl_filtered --seed=0 --save=100ep --epoch=100
# fine-tune
datasets=("bbbp" "tox21" "toxcast" "sider" "clintox" "muv" "hiv" "bace")
for dataset in ${datasets[*]}
do
    bash run_chem_rep.sh ${dataset}
done