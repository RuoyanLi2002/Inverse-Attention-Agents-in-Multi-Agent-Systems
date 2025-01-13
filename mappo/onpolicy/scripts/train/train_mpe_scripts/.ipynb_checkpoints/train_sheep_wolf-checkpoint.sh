#!/bin/sh
env="MPE"
scenario="simple_lineup_onlyfood_withoutcredit_humanshape" 
num_landmarks=0
num_agents=6
num_adv=3
algo="ippo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python /root/mappo/onpolicy/scripts/train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed 0 --num_adv ${num_adv}\
    --n_training_threads 1 --n_rollout_threads 64 --num_mini_batch 1 --episode_length 200 --num_env_steps 40000000\
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --share_policy False --wandb_name "Ruoyan Li" --user_name "ruoyanli" --attention 0 --shape_reward True
done
