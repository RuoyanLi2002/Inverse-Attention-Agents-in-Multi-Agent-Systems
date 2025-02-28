from gym import spaces
import numpy as np
import torch
from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

class Runner(object):
    def __init__(self, config):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps # 10e6
        self.episode_length = self.all_args.episode_length # 200
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.all_args.compete = False

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir
        
        self.scenario = self.all_args.scenario_name
        self.num_wolf = self.all_args.num_adv
        
    def initialize_policy(self, policy):
        self.policy = []
        self.trainer = []

        observation_space = []
        share_observation_space = []
        share_obs_dim = 0
        for i, agent in enumerate(self.envs.env.get_all_agents()):
            obs_dim = len(self.envs.env._get_obs(agent))
            share_obs_dim += obs_dim
            observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))  # [-inf,inf]
        
        share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.num_agents)]
        
        for agent_id in range(self.num_agents):
            # policy network
            
            po = Policy(self.all_args,
                    observation_space[agent_id],
                    share_observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    device = self.device,
                    agent_num = agent_id,
                    policy_type = policy[agent_id],)
            self.policy.append(po)

        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)

            bu = SeparatedReplayBuffer(self.all_args,
                                       observation_space[agent_id],
                                       share_observation_space[agent_id],
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)

    def restore(self, policy, agent_index):
        for agent_id in range(self.num_agents):
            if self.scenario == "simple_lineup_onlyfood_withoutcredit_humanshape" or self.scenario == "simple_sheep_wolf_landmark":
                agent_type = "WOLF" if agent_id < self.num_wolf else "SHEEP"
                policy_actor_state_dict = torch.load(f'/root/mappo_LQ/onpolicy/Actors/{policy[agent_id]}_{agent_type}_{agent_index[agent_id]}.pt')
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                # print(f"AGENT{agent_id} LOAD: {policy_actor_state_dict.keys()}")
                # if policy[agent_id] == "TOM":
                #     print(self.policy[agent_id].actor.imagined_we.fc.weight)

                print(f"Agent{agent_id} load {policy[agent_id]}_{agent_type}_{agent_index[agent_id]}.pt")
            elif self.scenario == "simple_landmark":
                policy_actor_state_dict = torch.load(f'/root/mappo_LQ/onpolicy/Actors/{policy[agent_id]}_{agent_index[agent_id]}.pt')
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)

                print(f"Agent{agent_id} load {policy[agent_id]}_{agent_index[agent_id]}.pt")