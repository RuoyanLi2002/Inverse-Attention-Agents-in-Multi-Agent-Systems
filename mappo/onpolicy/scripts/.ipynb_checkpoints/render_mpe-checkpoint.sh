#!/bin/sh
env="MPE"
scenario="simple_lineup_onlyfood_withoutcredit_humanshape"
num_landmarks=0
num_agents=6
num_adv=3
algo="mappo"
exp="check"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    # --use_recurrent_policy False --use_naive_recurrent_policy False
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs --share_policy False --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_adv ${num_adv} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 200 --render_episodes 2 --use_recurrent_policy False --all_args.use_naive_recurrent_policy False --attention 0 --model_dir "/home/vi3850-64core1/ruoyanli/mappo_LQ/onpolicy/scripts/results/MPE/simple_lineup_onlyfood_withoutcredit_humanshape/mappo/check/wandb/run-20240330_005436-koex9yno/files"
done
