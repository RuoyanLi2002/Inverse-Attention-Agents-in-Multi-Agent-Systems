#!/bin/sh
env="MPE"
scenario="simple_alltogether_observescore_noshape" 
num_landmarks=0
num_agents=6
num_adv=0
algo="mappo" #"mappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
# --wandb_name "longqian18" --user_name "longqian18"
# --use_wandb
# simple_alltogether_observescore_noshape
# simple_cluster3_credit_eva
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python ../eval/eval_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 100 --n_eval_rollout_threads 100 --episode_length 25 --num_env_steps 25000 --model_dir "../results/MPE/simple_alltogether_observescore_noshape/mappo/check/wandb/latest-run/files"\
    --share_policy False --use_wandb --attention 0 --num_adv ${num_adv}
done