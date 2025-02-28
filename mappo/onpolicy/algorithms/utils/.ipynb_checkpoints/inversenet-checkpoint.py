import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class InverseNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, dk, agent_num, seed):
        super(InverseNet, self).__init__()
        self.action_fc1 = torch.nn.Linear(5, hidden_dim)
        self.action_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.wolf_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.wolf_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.sheep_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.sheep_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.wall_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.wall_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.relu = F.relu

        self.dk = dk
        self.fc_query = torch.nn.Linear(hidden_dim, self.dk)
        self.fc_key = torch.nn.Linear(hidden_dim, self.dk)
        self.fc_value = torch.nn.Linear(hidden_dim, hidden_dim)
    
        # checkpoint = torch.load(f'/root/mappo_LQ/onpolicy/sheep_wolf44/InverseNet/inverse_net_{seed}_{agent_num}.pth', map_location=torch.device('cpu'))
        # model_state_dict = self.state_dict()
        # adjusted_state_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict}
        # self.load_state_dict(adjusted_state_dict, strict=True)
        # print(f"AGENT{agent_num} InverseNet IS LOAD: {adjusted_state_dict.keys()}")

        # print(f"AGENT{agent_num} INVERSENET IS NOT TRAIN")
        # for param in self.parameters():
        #     param.requires_grad = False
    
    def forward(self, action, wolf_gf, sheep_gf, wall_gf):
        action_embed = self.action_fc1(action)
        action_embed = self.relu(action_embed)
        action_embed = self.action_fc2(action_embed)
        action_embed = self.relu(action_embed)

        wolf_gf_embed = self.wolf_fc1(wolf_gf)
        wolf_gf_embed = self.relu(wolf_gf_embed)
        wolf_gf_embed = self.wolf_fc2(wolf_gf_embed)
        wolf_gf_embed = self.relu(wolf_gf_embed)

        sheep_gf_embed = self.sheep_fc1(sheep_gf)
        sheep_gf_embed = self.relu(sheep_gf_embed)
        sheep_gf_embed = self.sheep_fc2(sheep_gf_embed)
        sheep_gf_embed = self.relu(sheep_gf_embed)

        wall_gf_embed = self.wall_fc1(wall_gf)
        wall_gf_embed = self.relu(wall_gf_embed)
        wall_gf_embed = self.wall_fc2(wall_gf_embed)
        wall_gf_embed = self.relu(wall_gf_embed)

        obs_embed = torch.cat((torch.unsqueeze(action_embed, 1), wolf_gf_embed, sheep_gf_embed, torch.unsqueeze(wall_gf_embed, 1)), dim = 1)

        query = self.fc_query(obs_embed)
        key = self.fc_key(obs_embed)
        value = self.fc_value(obs_embed)

        weight = F.softmax(torch.bmm(query, torch.transpose(key, 1, 2)) / math.sqrt(self.dk), 2)
        weighted_gf = torch.bmm(weight, value)

        weight = weight[:, 0, :]
        weighted_gf = weighted_gf[:, 0, :]

        return weight, weighted_gf, value
    
class InverseNet_Landmark(nn.Module):
    def __init__(self, input_dim, hidden_dim, dk, agent_num, seed):
        super(InverseNet_Landmark, self).__init__()
        self.action_fc1 = torch.nn.Linear(5, hidden_dim)
        self.action_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.agent_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.agent_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.landmark_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.landmark_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.wall_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.wall_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.relu = F.relu

        self.dk = dk
        self.fc_query = torch.nn.Linear(hidden_dim, self.dk)
        self.fc_key = torch.nn.Linear(hidden_dim, self.dk)
        self.fc_value = torch.nn.Linear(hidden_dim, hidden_dim)

        # checkpoint = torch.load(f'/mnt/shared/qian/ruoyanli/mappo_LQ/onpolicy/landmark4/InverseNet/inverse_net_{seed}_{agent_num}.pth', map_location=torch.device('cpu'))
        # model_state_dict = self.state_dict()
        # adjusted_state_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict}
        # self.load_state_dict(adjusted_state_dict, strict=True)
        # print(f"AGENT{agent_num} InverseNet IS LOAD: {adjusted_state_dict.keys()}")

        # print(f"AGENT{agent_num} INVERSENET IS NOT TRAIN")
        # for param in self.parameters():
        #     param.requires_grad = False
    
    def forward(self, action, agent_gf, landmark_gf, wall_gf):
        action_embed = self.action_fc1(action)
        action_embed = self.relu(action_embed)
        action_embed = self.action_fc2(action_embed)
        action_embed = self.relu(action_embed)

        agent_gf_embed = self.agent_fc1(agent_gf)
        agent_gf_embed = self.relu(agent_gf_embed)
        agent_gf_embed = self.agent_fc2(agent_gf_embed)
        agent_gf_embed = self.relu(agent_gf_embed)

        landmark_gf_embed = self.landmark_fc1(landmark_gf)
        landmark_gf_embed = self.relu(landmark_gf_embed)
        landmark_gf_embed = self.landmark_fc2(landmark_gf_embed)
        landmark_gf_embed = self.relu(landmark_gf_embed)

        wall_gf_embed = self.wall_fc1(wall_gf)
        wall_gf_embed = self.relu(wall_gf_embed)
        wall_gf_embed = self.wall_fc2(wall_gf_embed)
        wall_gf_embed = self.relu(wall_gf_embed)

        obs_embed = torch.cat((torch.unsqueeze(action_embed, 1), agent_gf_embed, landmark_gf_embed, torch.unsqueeze(wall_gf_embed, 1)), dim = 1)

        query = self.fc_query(obs_embed)
        key = self.fc_key(obs_embed)
        value = self.fc_value(obs_embed)

        weight = F.softmax(torch.bmm(query, torch.transpose(key, 1, 2)) / math.sqrt(self.dk), 2)
        weighted_gf = torch.bmm(weight, value)

        weight = weight[:, 0, :]
        weighted_gf = weighted_gf[:, 0, :]

        return weight, weighted_gf, value
    
class InverseNet_sheep_wolf_landmark(nn.Module):
    def __init__(self, input_dim, hidden_dim, dk, agent_num, seed):
        super(InverseNet_sheep_wolf_landmark, self).__init__()
        self.action_fc1 = torch.nn.Linear(5, hidden_dim)
        self.action_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.wolf_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.wolf_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.sheep_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.sheep_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.wall_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.wall_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.landmark_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.landmark_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.relu = F.relu

        self.dk = dk
        self.fc_query = torch.nn.Linear(hidden_dim, self.dk)
        self.fc_key = torch.nn.Linear(hidden_dim, self.dk)
        self.fc_value = torch.nn.Linear(hidden_dim, hidden_dim)

        checkpoint = torch.load(f'/root/mappo_LQ/onpolicy/sheep_wolf_landmark4/InverseNet/inverse_net_{seed}_{agent_num}.pth', map_location=torch.device('cpu'))
        model_state_dict = self.state_dict()
        adjusted_state_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict}
        self.load_state_dict(adjusted_state_dict, strict=True)
        print(f"AGENT{agent_num} InverseNet IS LOAD: {adjusted_state_dict.keys()}")

        print(f"AGENT{agent_num} INVERSENET IS NOT TRAIN")
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, action, wolf_gf, sheep_gf, landmark_gf, wall_gf):
        action_embed = self.action_fc1(action)
        action_embed = self.relu(action_embed)
        action_embed = self.action_fc2(action_embed)
        action_embed = self.relu(action_embed)

        wolf_gf_embed = self.wolf_fc1(wolf_gf)
        wolf_gf_embed = self.relu(wolf_gf_embed)
        wolf_gf_embed = self.wolf_fc2(wolf_gf_embed)
        wolf_gf_embed = self.relu(wolf_gf_embed)

        sheep_gf_embed = self.sheep_fc1(sheep_gf)
        sheep_gf_embed = self.relu(sheep_gf_embed)
        sheep_gf_embed = self.sheep_fc2(sheep_gf_embed)
        sheep_gf_embed = self.relu(sheep_gf_embed)

        landmark_gf_embed = self.landmark_fc1(landmark_gf)
        landmark_gf_embed = self.relu(landmark_gf_embed)
        landmark_gf_embed = self.landmark_fc2(landmark_gf_embed)
        landmark_gf_embed = self.relu(landmark_gf_embed)

        wall_gf_embed = self.wall_fc1(wall_gf)
        wall_gf_embed = self.relu(wall_gf_embed)
        wall_gf_embed = self.wall_fc2(wall_gf_embed)
        wall_gf_embed = self.relu(wall_gf_embed)

        obs_embed = torch.cat((torch.unsqueeze(action_embed, 1), wolf_gf_embed, sheep_gf_embed, landmark_gf_embed, torch.unsqueeze(wall_gf_embed, 1)), dim = 1)

        query = self.fc_query(obs_embed)
        key = self.fc_key(obs_embed)
        value = self.fc_value(obs_embed)

        weight = F.softmax(torch.bmm(query, torch.transpose(key, 1, 2)) / math.sqrt(self.dk), 2)
        weighted_gf = torch.bmm(weight, value)

        weight = weight[:, 0, :]
        weighted_gf = weighted_gf[:, 0, :]

        return weight, weighted_gf, value