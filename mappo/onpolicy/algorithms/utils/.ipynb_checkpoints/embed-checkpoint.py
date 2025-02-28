import torch
import torch.nn as nn
import torch.nn.functional as F

class Embed_sheep_wolf(nn.Module):
    def __init__(self, input_dim, hidden_dim, agent_num, train_module, seed):
        super(Embed_sheep_wolf, self).__init__()
        self.velocity_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.velocity_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.wolf_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.wolf_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.sheep_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.sheep_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.wall_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.wall_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.relu = F.relu

        if not train_module:
            print(f"AGENT{agent_num} EMBED IS NOT TRAIN")
            for param in self.parameters():
                param.requires_grad = False
        else:
            print(f"AGENT{agent_num} EMBED IS TRAIN")

    def forward(self, velocity, wolf_gf, sheep_gf, wall_gf):
        velocity_embed = self.velocity_fc1(velocity)
        velocity_embed = self.relu(velocity_embed)
        velocity_embed = self.velocity_fc2(velocity_embed)
        velocity_embed = self.relu(velocity_embed)

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

        return velocity_embed, wolf_gf_embed, sheep_gf_embed, wall_gf_embed

class Embed_landmark(nn.Module):
    def __init__(self, input_dim, hidden_dim, agent_num, train_module, seed):
        super(Embed_landmark, self).__init__()
        self.velocity_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.velocity_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.agent_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.agent_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.landmark_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.landmark_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.wall_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.wall_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.relu = F.relu

        if not train_module:
            print(f"AGENT{agent_num} EMBED IS NOT TRAIN")
            for param in self.parameters():
                param.requires_grad = False
        else:
            print(f"AGENT{agent_num} EMBED IS TRAIN")

    def forward(self, velocity, agent_gf, landmark_gf, wall_gf):
        velocity_embed = self.velocity_fc1(velocity)
        velocity_embed = self.relu(velocity_embed)
        velocity_embed = self.velocity_fc2(velocity_embed)
        velocity_embed = self.relu(velocity_embed)

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

        return velocity_embed, agent_gf_embed, landmark_gf_embed, wall_gf_embed

class Embed_sheep_wolf_landmark(nn.Module):
    def __init__(self, input_dim, hidden_dim, agent_num, train_module, seed):
        super(Embed_sheep_wolf_landmark, self).__init__()
        self.velocity_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.velocity_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.wolf_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.wolf_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.sheep_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.sheep_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.landmark_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.landmark_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.wall_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.wall_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.relu = F.relu

        if not train_module:
            print(f"AGENT{agent_num} EMBED IS NOT TRAIN")
            for param in self.parameters():
                param.requires_grad = False
        else:
            print(f"AGENT{agent_num} EMBED IS TRAIN")

    def forward(self, velocity, wolf_gf, sheep_gf, landmark_gf, wall_gf):
        velocity_embed = self.velocity_fc1(velocity)
        velocity_embed = self.relu(velocity_embed)
        velocity_embed = self.velocity_fc2(velocity_embed)
        velocity_embed = self.relu(velocity_embed)

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

        return velocity_embed, wolf_gf_embed, sheep_gf_embed, landmark_gf_embed, wall_gf_embed