#!/usr/bin/env python
import random
import sys
import numpy as np
from pathlib import Path

import torch

from onpolicy.config import get_config

from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.env_wrappers import DummyVecEnv

def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MPEEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    return DummyVecEnv([get_env_fn(0)])

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_lineup_onlyfood_withoutcredit_humanshape', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=0)
    parser.add_argument("--num_adv", type=int, default=3)
    parser.add_argument("--attention", type=int, default=0)
    parser.add_argument("--compete", type=int, default=0)
    parser.add_argument('--num_agents', type=int,
                        default=6, help="number of players")
    parser.add_argument("--shape_reward", action='store_true', default=False)
                        

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # all_args.algorithm_name == "mappo":
    print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
    all_args.use_recurrent_policy = False 
    all_args.use_naive_recurrent_policy = False
    
    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    assert not (all_args.model_dir == None or all_args.model_dir == ""), ("set model_dir first")
    

    print("choose to use cpu...")
    device = torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    
    # env init
    envs = make_render_env(all_args)
    eval_envs = envs
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device
    }


    from onpolicy.runner.cross_comp.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    print("all_args.num_env_steps",all_args.num_env_steps)
    runner.eval(all_args.num_env_steps)
    
    # post process
    envs.close()

if __name__ == "__main__":
    main(sys.argv[1:])
