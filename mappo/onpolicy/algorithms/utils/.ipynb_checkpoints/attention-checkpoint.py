import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBase(nn.Module):
    def __init__(self, dk, hidden_dim, agent_num, train_module, seed):
        super(AttentionBase, self).__init__()
        self.dk = dk
        self.fc_query = torch.nn.Linear(hidden_dim, self.dk)
        self.fc_key = torch.nn.Linear(hidden_dim, self.dk)
        self.fc_value = torch.nn.Linear(hidden_dim, hidden_dim)
            
        if not train_module:
            print(f"AGENT{agent_num} ATTENTION IS NOT TRAIN")
            for param in self.parameters():
                param.requires_grad = False
        else:
            print(f"AGENT{agent_num} ATTENTION IS TRAIN")

    def forward(self, obs_embed):
        query = self.fc_query(obs_embed)
        key = self.fc_key(obs_embed)
        value = self.fc_value(obs_embed)

        weight = F.softmax(torch.bmm(query, torch.transpose(key, 1, 2)) / math.sqrt(self.dk), 2)
        weighted_gf = torch.bmm(weight, value)

        weight = weight[:, 0, :]
        weighted_gf = weighted_gf[:, 0, :]

        return weight, weighted_gf, value