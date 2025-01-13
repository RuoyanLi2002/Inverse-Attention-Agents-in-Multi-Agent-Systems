import time
import numpy as np
import torch
import pickle
import functools
import io
from torch_geometric.nn import knn_graph
# from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
# from functools import partial
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio
import sys
sys.path.append('/home/qian/TarGF/')
# from Algorithms.BallSDE import marginal_prob_std, diffusion_coeff, pc_sampler_state, ode_sampler, Euler_Maruyama_sampler
device='cpu'

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class TargetScore_update:
    def __init__(self, score, num_objs, max_vel, sigma=25):
        self.score = score
        self.num_objs = num_objs
        self.max_vel = max_vel
        # self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        # self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    def get_score(self, state_inp, t0, is_numpy=True, is_norm=True, empty=False):
        """
        state_inp: [bs, 3*num_objs]
        """
        if not empty:
            # construct graph-input for score network
            if not torch.is_tensor(state_inp):
                state_inp = torch.tensor(state_inp)
            positions = state_inp.view(-1, self.num_objs, 3).to(device)[:, :, :2]
            categories = state_inp.view(-1, self.num_objs, 3).to(device)[:, :, -1:]
            bs = positions.shape[0]
            positions = positions.view(-1, 2).float()
            categories = categories.view(-1).long()
            samples_batch = torch.tensor([i for i in range(bs) for _ in range(self.num_objs)], dtype=torch.int64).to(device)
            edge_index = knn_graph(positions, k=self.num_objs-1, batch=samples_batch)
            t = torch.tensor([t0]*bs).unsqueeze(1).to(device)
            inp_data = Data(x=positions, edge_index=edge_index, c=categories)

            out_score = self.score(inp_data, t, self.num_objs)
            out_score = out_score.detach()
            # normalise the gradient
            if is_norm:
                out_score = out_score * torch.min(
                    torch.tensor([1, self.max_vel / (torch.max(torch.abs(out_score)) + 1e-7)]).to(device))
            else:
                out_score = out_score
        else:
            out_score = torch.zeros_like(state_inp).to(device).view(-1, 2)
        return out_score.cpu().numpy() if is_numpy else out_score


class TargetScore:
    def __init__(self, score, num_objs, max_vel,sigma=25, sampler='ode', is_state=True):
        self.score = score
        self.is_state = is_state
        self.num_objs = num_objs
        self.max_vel = max_vel
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
        self.sampler_dict = {
            'pc': pc_sampler_state,
            'ode': ode_sampler, 
            'em': Euler_Maruyama_sampler,
        }
        self.sampler = self.sampler_dict[sampler]

    
    def get_score_gnn(self, state_inp, t0, is_numpy=True, is_norm=True, empty=False):
        if not empty:
            if self.is_state:
                if not torch.is_tensor(state_inp):
                    state_inp = torch.tensor(state_inp)
                state_inp = state_inp.view(-1, self.num_objs * 3, 2).to(device)
                bs = state_inp.shape[0]
                samples_batch = torch.tensor([i for i in range(bs) for _ in range(self.num_objs*3)], dtype=torch.int64).to(device)
                edge_index = knn_graph(state_inp.view(-1, 2), k=self.num_objs*3-1, batch=samples_batch)
                t = torch.tensor([t0]*bs).unsqueeze(1).to(device)
                inp_data = Data(x=state_inp.view(-1, 2), edge_index=edge_index).to(device)

                out_score = self.score(inp_data, t, self.num_objs) 
                out_score = out_score.detach()
                if is_norm:
                    out_score = out_score * torch.min(
                        torch.tensor([1, self.max_vel / (torch.max(torch.abs(out_score)) + 1e-7)]).to(device))
                else:
                    out_score = out_score
            else:
                state_ts = torch.FloatTensor(state_inp).to(device).unsqueeze(0)
                assert -1 <= torch.min(state_ts) <= torch.max(state_ts) <= 1
                t = torch.FloatTensor([t0]).to(device)
                out_score = self.score(state_ts, t)
                out_score = out_score.detach()
        else:
            out_score = torch.zeros_like(state_inp).to(device).view(-1, 2)
        return out_score.cpu().numpy() if is_numpy else out_score
    
    
    
    def get_score(self, state_inp, t0, is_numpy=True, is_norm=True, empty=False):
        if not empty:
            if self.is_state:
                if not torch.is_tensor(state_inp):
                    state_inp = torch.tensor(state_inp)
                dim = 4
                
                state_inp = state_inp.view(-1, self.num_objs * 3, 2).to(device)
                bs = state_inp.shape[0]
                # print(state_inp.shape)
                if True:
                    perturbed_x2 = torch.ones((state_inp.shape[0],state_inp.shape[1],4))
                    for i in range(int(bs)):
                        perturbed_x2[i][2][:2]=state_inp[i][0][:]-state_inp[i][0+2][:]
                        perturbed_x2[i][2][2:4]=state_inp[i][0+1][:]-state_inp[i][0+2][:]
                        perturbed_x2[i][0][:2]=state_inp[i][0+1][:]-state_inp[i][0][:]
                        perturbed_x2[i][0][2:4]=state_inp[i][0+2][:]-state_inp[i][0][:]
                        perturbed_x2[i][0+1][:2]=state_inp[i][0+0][:]-state_inp[i][0+1][:]
                        perturbed_x2[i][0+1][2:4]=state_inp[i][0+2][:]-state_inp[i][0+1][:]
                    
                    state_inp=perturbed_x2
                    state_inp = state_inp.to(device)

                    
                samples_batch = torch.tensor([i for i in range(bs) for _ in range(self.num_objs*3)], dtype=torch.int64).to(device)
                
                # edge_index = knn_graph(state_inp.view(-1, dim), k=self.num_objs*3-1, batch=samples_batch)
                
                t = torch.tensor([t0]*bs).unsqueeze(1).to(device)
                # inp_data = Data(x=state_inp.view(-1, dim), edge_index=edge_index)
                inp_data = Data(x=state_inp.view(-1, dim))
                out_score = self.score(inp_data, t, self.num_objs) 
                out_score = out_score.detach()
                if is_norm:
                    out_score = out_score * torch.min(
                        torch.tensor([1, self.max_vel / (torch.max(torch.abs(out_score)) + 1e-7)]).to(device))
                else:
                    out_score = out_score
            else:
                state_ts = torch.FloatTensor(state_inp).to(device).unsqueeze(0)
                assert -1 <= torch.min(state_ts) <= torch.max(state_ts) <= 1
                t = torch.FloatTensor([t0]).to(device)
                out_score = self.score(state_ts, t)
                out_score = out_score.detach()
        else:
            out_score = torch.zeros_like(state_inp).to(device).view(-1, 2)
        return out_score.cpu().numpy() if is_numpy else out_score

    def get_score_reference2(self, state_inp, t0, is_numpy=True, is_norm=False, empty=False):
        if not empty:
            if self.is_state:
                if not torch.is_tensor(state_inp):
                    state_inp = torch.tensor(state_inp)
                dim = 2
                
                state_inp = state_inp.view(-1, self.num_objs * 3, 2).to(device)
                bs = state_inp.shape[0]
                # print(state_inp.shape)
                if True:
                    perturbed_x2 = torch.ones((state_inp.shape[0],state_inp.shape[1],dim))
                    for i in range(int(bs)):
                        perturbed_x2[i][2][:2]=state_inp[i][0+2][:]
                        perturbed_x2[i][0][:2]=state_inp[i][0+1][:]-state_inp[i][0][:]
                        perturbed_x2[i][1][:2]=state_inp[i][0+0][:]-state_inp[i][0+1][:]
                    
                    state_inp=perturbed_x2
                    state_inp = state_inp.to(device)

                    
                samples_batch = torch.tensor([i for i in range(bs) for _ in range(self.num_objs*3)], dtype=torch.int64).to(device)
                
                # edge_index = knn_graph(state_inp.view(-1, dim), k=self.num_objs*3-1, batch=samples_batch)
                
                t = torch.tensor([t0]*bs).unsqueeze(1).to(device)
                # inp_data = Data(x=state_inp.view(-1, dim), edge_index=edge_index)
                inp_data = Data(x=state_inp.view(-1, dim))
                # print(inp_data.x.shape,'shape!!!!!!!!!')
                out_score = self.score(inp_data, t, self.num_objs) 
                out_score = out_score.detach()
                if is_norm:
                    out_score = out_score * torch.min(
                        torch.tensor([1, self.max_vel / (torch.max(torch.abs(out_score)) + 1e-7)]).to(device))
                else:
                    out_score = out_score
            else:
                state_ts = torch.FloatTensor(state_inp).to(device).unsqueeze(0)
                assert -1 <= torch.min(state_ts) <= torch.max(state_ts) <= 1
                t = torch.FloatTensor([t0]).to(device)
                out_score = self.score(state_ts, t)
                out_score = out_score.detach()
        else:
            out_score = torch.zeros_like(state_inp).to(device).view(-1, 2)
        return out_score.cpu().numpy() if is_numpy else out_score

    def get_score2(self, state_inp, t0, n_box, is_numpy=True, is_norm=True, empty=False):
        # print("nbox!!!!!!!!!!!",n_box)
        if not empty:
            if self.is_state:
                if not torch.is_tensor(state_inp):
                    state_inp = torch.tensor(state_inp)
                dim = 2*(n_box*3-1)
                
                state_inp = state_inp.view(-1, self.num_objs * 3, 2).to(device)
                bs = state_inp.shape[0]
                # print(state_inp.shape)
                if True:
                    perturbed_x2 = torch.ones((state_inp.shape[0],state_inp.shape[1],2*(n_box*3-1)))
                    for i in range(int(bs)):
                        for j in range(3*n_box):
                            start = 0
                            for m in range(3*n_box):
                                if m==j:
                                    continue
                                else:
                                    perturbed_x2[i][j][start:start+2]=state_inp[i][m][:]-state_inp[i][j][:]
                                    start += 2
                    
                    state_inp=perturbed_x2
                    state_inp = state_inp.to(device)

                    
                samples_batch = torch.tensor([i for i in range(bs) for _ in range(self.num_objs*3)], dtype=torch.int64).to(device)
                
                # edge_index = knn_graph(state_inp.view(-1, dim), k=self.num_objs*3-1, batch=samples_batch)
                
                t = torch.tensor([t0]*bs).unsqueeze(1).to(device)
                # inp_data = Data(x=state_inp.view(-1, dim), edge_index=edge_index)
                inp_data = Data(x=state_inp.view(-1, dim))
                out_score = self.score(inp_data, t, self.num_objs) 
                out_score = out_score.detach()
                if is_norm:
                    out_score = out_score * torch.min(
                        torch.tensor([1, self.max_vel / (torch.max(torch.abs(out_score)) + 1e-7)]).to(device))
                else:
                    out_score = out_score
            else:
                state_ts = torch.FloatTensor(state_inp).to(device).unsqueeze(0)
                assert -1 <= torch.min(state_ts) <= torch.max(state_ts) <= 1
                t = torch.FloatTensor([t0]).to(device)
                out_score = self.score(state_ts, t)
                out_score = out_score.detach()
        else:
            out_score = torch.zeros_like(state_inp).to(device).view(-1, 2)
        return out_score.cpu().numpy() if is_numpy else out_score
def fix(map_loc):
    # Closure rather than a lambda to preserve map_loc 
    return lambda b: torch.load(BytesIO(b), map_location=map_loc)

class MappedUnpickler(pickle.Unpickler):

    def __init__(self, *args, map_location='cpu', **kwargs):
        self._map_location = map_location
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return fix(self._map_location)
        else: 
            return super().find_class(module, name)

def mapped_loads(s, map_location='cpu'):
    bs = io.BytesIO(s)
    unpickler = MappedUnpickler(bs, map_location=map_location)
    return unpickler.load()
def load_target_score(num_objs, max_action ,is_state=True):
    # diffusion_coeff_func = functools.partial(diffusion_coeff, sigma=25)
    tar_path = '/home/qian/TarGF/logs/randomblue_mlp_refence4_score/score.pt'
    with open(tar_path, 'rb') as f:
        # score_target = pickle.load(f)
        score_target =  CPU_Unpickler(f).load()
    return TargetScore(score_target.to(device), num_objs, max_action, is_state=is_state)

def load_target_score2(num_objs, max_action ,is_state=True):
    # diffusion_coeff_func = functools.partial(diffusion_coeff, sigma=25)
    tar_path = '/home/qian/TarGF/logs/lineup_Score/score.pt'
    with open(tar_path, 'rb') as f:
        # score_target = pickle.load(f)
        score_target =  CPU_Unpickler(f).load()
    return TargetScore(score_target.to(device), num_objs, max_action, is_state=is_state)



def load_target_score3(num_objs, max_action ,is_state=True):
    # diffusion_coeff_func = functools.partial(diffusion_coeff, sigma=25)
    tar_path = '/home/qian/TarGF/logs/alltogether_6_refence2_mlp_score/score.pt'
    with open(tar_path, 'rb') as f:
        # score_target = mapped_loads(f, map_location='cpu')
        score_target =  CPU_Unpickler(f).load()
        # print(score_target.is_cuda(),'cuDAAAAAAAAAAAA')
    return TargetScore(score_target.to(device), num_objs, max_action, is_state=is_state)

def load_target_score4(num_objs, max_action ,is_state=True):
    # diffusion_coeff_func = functools.partial(diffusion_coeff, sigma=25)
    tar_path = '/home/qian/TarGF/logs/randomblue_mlp_refence2_score/score.pt'
    with open(tar_path, 'rb') as f:
        # score_target = pickle.load(f)
        score_target =  CPU_Unpickler(f).load()
    return TargetScore(score_target.to(device), num_objs, max_action, is_state=is_state)


def load_target_score5(num_objs, max_action ,is_state=True):
    # diffusion_coeff_func = functools.partial(diffusion_coeff, sigma=25)
    tar_path = '/home/qian/targfupdate/TarGF/logs/Lineup_onedirection_score/score.pt'
    with open(tar_path, 'rb') as f:
        # score_target = mapped_loads(f, map_location='cpu')
        score_target =  CPU_Unpickler(f).load()
        # print(score_target.is_cuda(),'cuDAAAAAAAAAAAA')
    return TargetScore_update(score_target.to(device), num_objs, max_action)

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MPERunner, self).__init__(config)

    def run(self):
        print("share start warmup")
        self.warmup()   
        print("share end warmup")

        print("testpytorch!!!!!!!!!!!!!!!!!")
        target_score = load_target_score(1, 0.3 ,is_state=True)
        state=np.zeros((1,6))
        scoretest = target_score.get_score(state, 0)
        print(scoretest,"scoretest")
        print("testpytorch?????????????????")

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                # if step==4:
                #     print(actions,'actions')
                #     print(actions_env,'actions_env')  
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
