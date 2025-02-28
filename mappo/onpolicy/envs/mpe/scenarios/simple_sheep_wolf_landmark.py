import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario
import random

import pickle
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import io
import sys
import functools

import os
from matplotlib import pyplot as plt

sys.path.append('/root/exp1/targfupdate/')
from Algorithms.BallSDE import marginal_prob_std, diffusion_coeff

obs = 0.5
# device = "cuda:0"
device = "cpu"

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
        
class TargetScore_Wall:
    def __init__(self, score, num_objs, max_vel, sigma=25):
        self.score = score
        self.num_objs = num_objs
        self.max_vel = max_vel
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    def get_score(self, x, t0, is_numpy=True, is_norm=True, empty=False):
        if not empty:
            if not torch.is_tensor(x):
                x = torch.tensor(x)
            x = x.to(device)
            bs = x.shape[0]
            t = torch.tensor([t0]*bs).unsqueeze(1).to(device)
            #  model(perturbed_x.float(), random_t, num_objs)
            out_score = self.score(x, t, self.num_objs)
            out_score = out_score.detach()

        else:
            out_score = torch.zeros_like(x).to(device).view(-1, 2)
        return out_score.cpu().numpy() if is_numpy else out_score
    
class TargetScore:
    def __init__(self, score, num_objs, max_vel, sigma=25):
        self.score = score
        self.num_objs = num_objs
        self.max_vel = max_vel
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    def get_score(self, state_inp, t0, is_numpy=True, is_norm=True, empty=False):
        if not empty:
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
            if is_norm:
                out_score = out_score * torch.min(
                    torch.tensor([1, self.max_vel / (torch.max(torch.abs(out_score)) + 1e-7)]).to(device))
            else:
                out_score = out_score
        else:
            out_score = torch.zeros_like(state_inp).to(device).view(-1, 2)
        return out_score.cpu().numpy() if is_numpy else out_score
        
class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        # set any world properties first
        world.dim_c = 2
        self.num_agents = args.num_agents
        self.num_good_agents = args.num_agents - args.num_adv
        self.num_adversaries = args.num_adv
        num_landmarks = args.num_landmarks
        self.use_shape_reward = args.shape_reward
        print(f"Wolf: {self.num_adversaries}; Sheep: {self.num_good_agents}; Landmark: {args.num_landmarks}; Use Shape Reward: {self.use_shape_reward}")
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < self.num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1 if agent.adversary else 1.3
            agent.visualize = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.02
            landmark.boundary = False
            landmark.occupy = 0
        self.num_food = 0
        self.reset_world(world)

        self.target_score, self.diffusion_coeff_fn = self.load_target_score(num_objs = 2, max_action = 1)
        self.target_score_wall, self.diffusion_coeff_fn_wall = self.load_target_score_wall(num_objs = 2, max_action = 1)
        
        self.gf_num = 0

        world.past_action = [np.random.rand(5).astype(np.float32) for _ in range(self.num_agents)]
        
        world.past_obs = [[np.random.standard_normal(2).tolist() for i in range(self.num_agents+num_landmarks)] for j in range(self.num_agents)]
        world.new_obs = [[] for _ in range(self.num_agents)]
        
        policy = "MAPPO" if args.policy == 0 else "GF+ATTENTION"
        world.ls_policy = [policy for i in range(self.num_agents)]
        return world

    def reset_world(self, world):
        # color for wolves
        for i in range(0, self.num_adversaries):
            world.agents[i].color = np.array([0.85, 0.35, 0.35])
        for i in range(self.num_adversaries, self.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
        
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def distance_between_agents(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        return dist

    def reward(self, agent, world):
        return self.reward_wolf(agent, world) if agent.adversary else self.reward_sheep(agent, world)

    def reward_sheep(self, agent, world):
        total_reward = 0
        for temp_agent in world.agents:
            if temp_agent is agent:
                continue
            elif self.is_collision(agent, temp_agent) and temp_agent.adversary:
                total_reward -= 5

        for landmark in world.landmarks:
            if self.is_collision(agent, landmark):
                total_reward += 2
                landmark.occupy = 1

        if self.use_shape_reward:
            dis_reward = min([np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos))) for landmark in world.landmarks])
            total_reward -= dis_reward/5

        return total_reward

    def reward_wolf(self, agent, world):
        total_reward = 0

        for temp_agent in world.agents:
            if temp_agent is agent:
                continue
            elif not temp_agent.adversary and self.is_collision(agent, temp_agent):
                total_reward += 5

        if self.use_shape_reward:
            dis_reward = min([self.distance_between_agents(agent, sheep) for sheep in self.good_agents(world)])
            total_reward -= 0.2 * dis_reward

        return total_reward
    
    def info(self, agent, world):
        if agent.adversary:
            rew = self.reward_wolf(agent, world)
        else:
            rew = self.reward_sheep(agent, world)

        return {'rew':rew}
    
    def obs_MAPPO(self, agent, world):
        # _ = self.obs_GFATTENTION(agent, world)

        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            
        landmark_pos = []
        for landmark in world.landmarks:
            landmark_pos.append(landmark.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + other_pos + landmark_pos)

    def obs_GFATTENTION(self, agent, world):
        agent_num = self.find_agent_num(agent, world)

        gf = [agent.state.p_vel]
        agent_pos = np.concatenate([agent.state.p_pos, np.array([1.0])]) if agent.adversary else np.concatenate([agent.state.p_pos, np.array([0.0])])

        world.new_obs[agent_num] = []
        for other in world.agents:
            if other is agent:
                continue
            else:
                other_pos = np.concatenate([other.state.p_pos, np.array([1.0])]) if other.adversary else np.concatenate([other.state.p_pos, np.array([0.0])])
                pos = np.concatenate([agent_pos, other_pos])
                
                temp_gf = self.get_agent_gradient_field(pos)
                temp_gf = temp_gf[0] if agent.adversary else -temp_gf[0]
                
                gf.append(temp_gf)
                world.new_obs[agent_num].append(temp_gf.tolist())
        
        for landmark in world.landmarks:
            landmark_pos = np.concatenate([landmark.state.p_pos, np.array([0.0])])

            pos = np.concatenate([agent_pos, landmark_pos])

            temp_gf = self.get_agent_gradient_field(pos)
            temp_gf = temp_gf[0]
                
            gf.append(temp_gf)
            world.new_obs[agent_num].append(temp_gf.tolist())
        
        wall_gf = self.get_wall_gradient_field(agent.state.p_pos)
        gf.append(wall_gf)
        world.new_obs[agent_num].append(wall_gf.tolist())
            
        gf = np.array(gf)

        return gf.reshape(-1)

    def obs_TOM(self, agent, world):
        agent_num = self.find_agent_num(agent, world)

        gf = [agent.state.p_vel]
        agent_pos = np.concatenate([agent.state.p_pos, np.array([1.0])]) if agent.adversary else np.concatenate([agent.state.p_pos, np.array([0.0])])

        world.new_obs[agent_num] = []
        for other in world.agents:
            if other is agent:
                continue
            else:
                other_pos = np.concatenate([other.state.p_pos, np.array([1.0])]) if other.adversary else np.concatenate([other.state.p_pos, np.array([0.0])])
                pos = np.concatenate([agent_pos, other_pos])
                
                temp_gf = self.get_agent_gradient_field(pos)
                temp_gf = temp_gf[0] if agent.adversary else -temp_gf[0]
                
                gf.append(temp_gf)
                world.new_obs[agent_num].append(temp_gf.tolist())
        
        for landmark in world.landmarks:
            landmark_pos = np.concatenate([landmark.state.p_pos, np.array([0.0])])

            pos = np.concatenate([agent_pos, landmark_pos])

            temp_gf = self.get_agent_gradient_field(pos)
            temp_gf = temp_gf[0]
                
            gf.append(temp_gf)
            world.new_obs[agent_num].append(temp_gf.tolist())
        
        wall_gf = self.get_wall_gradient_field(agent.state.p_pos)
        gf.append(wall_gf)
        world.new_obs[agent_num].append(wall_gf.tolist())

        for other1 in world.agents:
            if other1 is agent:
                continue
            elif self.same_team(agent, other1):
                other1_num = self.find_agent_num(other1, world)
                gf = self.append_actions(gf, other1, world)
                gf = gf + world.past_obs[other1_num]
            
        gf = np.array(gf)

        return gf.reshape(-1)
    
    def observation(self, agent, world):
        agent_num = self.find_agent_num(agent, world)
        policy = world.ls_policy[agent_num]
        if policy == "MAPPO":
            return self.obs_MAPPO(agent, world)
        elif policy == "GF+ATTENTION":
            return self.obs_MAPPO(agent, world)
        elif policy == "TOM":
            return self.obs_TOM(agent, world)
        
    
    
    def get_agent_gradient_field(self, pos):
        obs = torch.tensor(pos)
        gf = self.target_score.get_score(obs, t0 = 0.01, is_norm=False)

        return gf
    
    def get_wall_gradient_field(self, pos):
        obs = torch.tensor(pos).to(torch.float32)
        # print(obs)
        # print(self.target_score_wall.score.l1[0].weight.data.dtype)
        gf = self.target_score_wall.get_score(obs, t0 = 0.01, is_norm=False)

        return gf[0]

    def load_target_score(self, num_objs, max_action):
        diffusion_coeff_func = functools.partial(diffusion_coeff, sigma=25)
        tar_path = '/root/exp1/targfupdate/score_wolf2.pt'
        with open(tar_path, 'rb') as f:
            # score_target = pickle.load(f)
            score_target = CPU_Unpickler(f).load()
        return TargetScore(score_target.to(device), num_objs, max_action), diffusion_coeff_func
    
    def load_target_score_wall(self, num_objs, max_action):
        diffusion_coeff_func = functools.partial(diffusion_coeff, sigma=25)
        tar_path = '/root/exp1/targfupdate/score_wall2_fc.pt'
        with open(tar_path, 'rb') as f:
            # score_target = pickle.load(f)
            score_target = CPU_Unpickler(f).load()
        return TargetScore_Wall(score_target.to(device), num_objs, max_action), diffusion_coeff_func
    
    def find_agent_num(self, agent, world):
        count = 0
        for other in world.agents:
            if other is agent:
                return count
            else:
                count += 1
        
        return None
    
    def append_actions(self, gf, agent, world):
        agent_num = self.find_agent_num(agent, world)
        action = list(world.past_action[agent_num])
        gf.append(action[0:2])
        gf.append(action[2:4])
        gf.append([action[4], 0.0])

        return gf

    def same_team(self, agent1, agent2):
        return agent1.adversary == agent2.adversary