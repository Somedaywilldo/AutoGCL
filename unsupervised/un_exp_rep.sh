for seed in {1..4}
do
    python us_main.py --dataset=$1 --seed=${seed} --save='with_sim_loss' --gpu=$2 --exp='cl_exp' --lr=0.001 --hidden-dim=128 --epochs=30
done