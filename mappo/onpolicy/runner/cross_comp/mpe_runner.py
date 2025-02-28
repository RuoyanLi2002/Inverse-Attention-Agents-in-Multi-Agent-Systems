import time
import imageio
import random
from tqdm import trange
import numpy as np
import torch
from onpolicy.runner.cross_comp.base_runner import Runner

device='cpu'

USE_BOX = 1

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)

    def sample_policy(self, num_agent):
        ls_policy = ["MAPPO", "GF+ATTENTION", "TOM"]
        # ls_policy = ["GF+ATTENTION", "TOM"]
        return random.choices(ls_policy, k=num_agent)
    
    def sample_agent(self, max_num = 2):
        return [random.randint(0, max_num) for _ in range(3)]

    @torch.no_grad()
    def eval(self, total_num_steps):
        if self.scenario == "simple_lineup_onlyfood_withoutcredit_humanshape" or self.scenario == "simple_sheep_wolf_landmark":
            agent_episode_reward = {"MAPPO_WOLF": [], "MAPPO_SHEEP": [],
                                    "GF+ATTENTION_WOLF": [],"GF+ATTENTION_SHEEP": [],
                                    "TOM_WOLF": [], "TOM_SHEEP": []}
        elif self.scenario == "simple_landmark":
            print("------")
            agent_episode_reward = {"MAPPO": [], "GF+ATTENTION": [], "TOM": []}
        
        all_frames = []
            
        self.all_args.render_episodes = 10**3
        self.episode_length = 200
        for episode in range(self.all_args.render_episodes):
            policy = self.sample_policy(self.num_agents)
            self.envs.env.world.ls_policy = policy
            print(self.envs.env.world.ls_policy)
            
            agent_index = self.sample_agent()
            self.initialize_policy(policy)
            self.restore(policy, agent_index)
            

            episode_rewards = []
            obs = self.eval_envs.reset()
            
            if self.all_args.save_gifs:
                image = self.envs.render(mode='rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            for step in range(self.episode_length):
                calc_start = time.time()        
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    action_env = action

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                   
                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.eval_envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
                
                if self.all_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)


            episode_rewards = np.array(episode_rewards)
            print(f"Episode: {episode}")
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))
                
                if self.scenario == "simple_lineup_onlyfood_withoutcredit_humanshape" or self.scenario == "simple_sheep_wolf_landmark":
                    agent_type = "WOLF" if agent_id < self.num_wolf else "SHEEP"
                    agent_episode_reward[f"{policy[agent_id]}_{agent_type}"].append(average_episode_rewards)
                elif self.scenario == "simple_landmark":
                    agent_episode_reward[f"{policy[agent_id]}"].append(average_episode_rewards)
                    
            print("--------------------")
        
        print("---------Final Value-----------")
        for agent_policy in agent_episode_reward:
            final_reward = np.mean(agent_episode_reward[agent_policy])
            print(f"{agent_policy} Final Reward: {final_reward}")
            
        if self.all_args.save_gifs:
            imageio.mimsave('render.gif', all_frames, duration=self.all_args.ifi)