env="MPE"
scenario="simple_sheep_wolf_landmark" 
num_landmarks=4
num_agents=6
num_adv=3
algo="mappo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python cross_comp.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --n_eval_rollout_threads 100 --episode_length 200 --num_env_steps 200000 --model_dir "/home/taogao/mappo_LQ/onpolicy/scripts/results/MPE/simple_lineup_onlyfood_withoutcredit_humanshape/mappo/check/wandb/run-20240223_141314-ake3s2mb/files"\
    --share_policy False --use_wandb --attention 0 --num_adv ${num_adv}
done