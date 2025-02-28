import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TOM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TOM, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = F.relu
        self.softmax = F.softmax

        self.temp_fc = torch.nn.Linear(input_dim, output_dim)
        # print("2 MLP + Linear")
        # print("USING TOM MLP")

    def forward(self, weight):
        weight = self.fc1(weight)
        weight = self.fc2(weight)
        
        return weight
    
class TOM_Initialization(nn.Module):
    def __init__(self, input_dim, output_dim, epsilon = 0.0001):
        super(TOM_Initialization, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim

        n = output_dim
        weight_matrix = torch.zeros(n, 4*n)
        weight_matrix[:n, :n] = torch.eye(n)
        self.fc.weight.data = weight_matrix
        self.fc.bias.data.fill_(0.0)

        print(f"USING TOM MLP Initialization: {self.fc.weight.data}")

    def forward(self, weight):
        weight = self.reshape(weight)
        weight = self.fc(weight)
        
        return weight

    def reshape(self, weight):
        batch_size = weight.shape[0]
        weight = weight.view(weight.shape[0], 4, self.output_dim)

        for i in range(batch_size):
            if torch.rand(1) < 0.5:
                weight[i, [1, 2]] = weight[i, [2, 1]]
            if torch.rand(1) < 0.5:
                weight[i, [2, 3]] = weight[i, [3, 2]]

        weight = weight.view(batch_size, -1)

        return weight
    
class TOM_Attention(nn.Module):
    def __init__(self, input_dim, dk):
        super(TOM_Attention, self).__init__()
        self.input_dim = input_dim
        self.dk = dk
        self.fc_query = torch.nn.Linear(input_dim, self.dk)
        self.fc_key = torch.nn.Linear(input_dim, self.dk)
        self.fc_value = torch.nn.Linear(input_dim, input_dim)

        print("USING TOM ATTENTION")

    def forward(self, weight):
        weight = weight.view(weight.shape[0], 3, self.input_dim)

        query = self.fc_query(weight)
        key = self.fc_key(weight)
        value = self.fc_value(weight)

        att_weight = F.softmax(torch.bmm(query, torch.transpose(key, 1, 2)) / math.sqrt(self.dk), 2)
        weighted_weight = torch.bmm(att_weight, value)

        return weighted_weight[:, 0, :]
    
class TOM_Attention_Rearrange(nn.Module):
    def __init__(self, input_dim, dk, agent_num):
        super(TOM_Attention_Rearrange, self).__init__()
        self.input_dim = input_dim
        self.dk = dk
        self.fc_query = torch.nn.Linear(input_dim, self.dk)
        self.fc_key = torch.nn.Linear(input_dim, self.dk)
        self.fc_value = torch.nn.Linear(input_dim, input_dim)

        self.agent_num = agent_num

        print("USING TOM ATTENTION Rearrange")

    def forward(self, weight):
        weight = weight.view(weight.shape[0], 3, self.input_dim)
        weight = self.transform_obs(weight)

        query = self.fc_query(weight)
        key = self.fc_key(weight)
        value = self.fc_value(weight)

        att_weight = F.softmax(torch.bmm(query, torch.transpose(key, 1, 2)) / math.sqrt(self.dk), 2)
        weighted_weight = torch.bmm(att_weight, value)

        return weighted_weight[:, 0, :]
    
    def transform_obs(self, weight):
        if self.agent_num == 0:
            weight[:,1,:] = weight[:,1, [1, 0, 2, 3, 4, 5, 6]]
            weight[:,2,:] = weight[:,2, [1, 2, 0, 3, 4, 5, 6]]
            return weight
        
        elif self.agent_num == 5:
            weight[:,1,:] = weight[:,1, [5, 1, 2, 3, 0, 4, 6]]
            weight[:,2,:] = weight[:,2, [5, 1, 2, 3, 4, 0, 6]]
            return weight
        
class TOM_Attention_Rearrange_Duplicate(nn.Module):
    def __init__(self, input_dim, dk, agent_num, n):
        super(TOM_Attention_Rearrange_Duplicate, self).__init__()
        self.input_dim = input_dim
        self.dk = dk
        self.fc_query = torch.nn.Linear(input_dim, self.dk)
        self.fc_key = torch.nn.Linear(input_dim, self.dk)
        self.fc_value = torch.nn.Linear(input_dim, input_dim)

        self.agent_num = agent_num
        self.n = n

        print("USING TOM ATTENTION Duplicate")

    def forward(self, weight):
        weight = weight.view(weight.shape[0], 3, self.input_dim)
        weight = self.transform_obs(weight)

        first_row = weight[:, 0, :].unsqueeze(1)
        repeated_first_row = first_row.repeat(1, self.n, 1)
        last_two_rows = weight[:, 1:, :]

        weight = torch.cat((repeated_first_row, last_two_rows), dim=1)

        query = self.fc_query(weight)
        key = self.fc_key(weight)
        value = self.fc_value(weight)

        att_weight = F.softmax(torch.bmm(query, torch.transpose(key, 1, 2)) / math.sqrt(self.dk), 2)
        weighted_weight = torch.bmm(att_weight, value)

        return weighted_weight[:, 0, :]
    
    def transform_obs(self, weight):
        if self.agent_num == 0:
            weight[:,1,:] = weight[:,1, [1, 0, 2, 3, 4, 5, 6]]
            weight[:,2,:] = weight[:,2, [1, 2, 0, 3, 4, 5, 6]]
            return weight
        
        elif self.agent_num == 5:
            weight[:,1,:] = weight[:,1, [5, 1, 2, 3, 0, 4, 6]]
            weight[:,2,:] = weight[:,2, [5, 1, 2, 3, 4, 0, 6]]
            return weight
        

class TOM_MLP_Value(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TOM_MLP_Value, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim

        n = output_dim
        weight_matrix = torch.zeros(n, 3*n)
        weight_matrix[:n, :n] = torch.eye(n)
        self.fc.weight.data = weight_matrix
        self.fc.bias.data.fill_(0.0)

        print(f"USING TOM MLP Value Initialization: {self.fc.weight.data}")

        # print(f"USING TOM MLP Value")

    def forward(self, my_weight, my_value, ally1_weight, ally1_value, ally2_weight, ally2_value):
        my_length = torch.norm(my_value, p=2, dim=2)
        my_adj_weight = my_weight*my_length

        ally1_length = torch.norm(ally1_value, p=2, dim=2)
        ally1_adj_weight = ally1_weight*ally1_length

        ally2_length = torch.norm(ally2_value, p=2, dim=2)
        ally2_adj_weight = ally2_weight*ally2_length

        weight = torch.cat((my_adj_weight, ally1_adj_weight, ally2_adj_weight), dim = 1)
        weight = self.reshape(weight)

        weight = self.fc(weight)

        weight = weight / my_length
        
        return weight
    
    def reshape(self, weight):
        batch_size = weight.shape[0]
        weight = weight.view(weight.shape[0], 3, self.output_dim)

        for i in range(batch_size):
            if torch.rand(1) < 0.5:
                weight[i, [1, 2]] = weight[i, [2, 1]]

        weight = weight.view(batch_size, -1)

        return weight
    
class TOM_ATTENTION_Value(nn.Module):
    def __init__(self, input_dim, dk, agent_num):
        super(TOM_ATTENTION_Value, self).__init__()
        self.input_dim = input_dim
        self.dk = dk
        self.fc_query = torch.nn.Linear(input_dim, self.dk)
        self.fc_key = torch.nn.Linear(input_dim, self.dk)
        self.fc_value = torch.nn.Linear(input_dim, input_dim)

        self.agent_num = agent_num

        print(f"USING TOM ATTENTION Value")

    def forward(self, my_weight, my_value, ally1_weight, ally1_value, ally2_weight, ally2_value):
        my_length = torch.norm(my_value, p=2, dim=2)
        my_adj_weight = my_weight*my_length

        ally1_length = torch.norm(ally1_value, p=2, dim=2)
        ally1_adj_weight = ally1_weight*ally1_length

        ally2_length = torch.norm(ally2_value, p=2, dim=2)
        ally2_adj_weight = ally2_weight*ally2_length

        weight = torch.cat((my_adj_weight, ally1_adj_weight, ally2_adj_weight), dim = 1)

        weight = weight.view(weight.shape[0], 3, self.input_dim)
        weight = self.transform_obs(weight)

        query = self.fc_query(weight)
        key = self.fc_key(weight)
        value = self.fc_value(weight)

        att_weight = F.softmax(torch.bmm(query, torch.transpose(key, 1, 2)) / math.sqrt(self.dk), 2)
        weighted_weight = torch.bmm(att_weight, value)

        return weighted_weight[:, 0, :]
    
    def transform_obs(self, weight):
        if self.agent_num == 0:
            weight[:,1,:] = weight[:,1, [1, 0, 2, 3, 4, 5, 6]]
            weight[:,2,:] = weight[:,2, [1, 2, 0, 3, 4, 5, 6]]
            return weight
        
        elif self.agent_num == 5:
            weight[:,1,:] = weight[:,1, [5, 1, 2, 3, 0, 4, 6]]
            weight[:,2,:] = weight[:,2, [5, 1, 2, 3, 4, 0, 6]]
            return weight
        