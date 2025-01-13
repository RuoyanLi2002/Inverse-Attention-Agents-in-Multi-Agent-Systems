from .distributions import DiagGaussian_new, DiagGaussian
import torch
import torch.nn as nn

class ACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, hidden_dim, agent_num, train_module, seed):
        super(ACTLayer, self).__init__()

        # print(action_space,'actionspace')
        # print("Box")
        action_dim = 5
        self.action_out = DiagGaussian_new(inputs_dim, hidden_dim, action_dim, use_orthogonal, gain)

        if not train_module:
            print(f"AGENT{agent_num} ACTLAYER IS NOT TRAIN")
            for param in self.parameters():
                param.requires_grad = False
        else:
            print(f"AGENT{agent_num} ACTLAYER IS TRAIN")

    def forward(self, x, available_actions=None, deterministic=False):
        action_logits = self.action_out(x)
        actions = action_logits.mode() if deterministic else action_logits.sample()
                
        action_log_probs = action_logits.log_probs(actions)
        
        return actions, action_log_probs

    def get_probs(self, x, available_actions=None):
        action_logits = self.action_out(x, available_actions)
        action_probs = action_logits.probs
        
        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        action_logits = self.action_out(x)
        action_log_probs = action_logits.log_probs(action)

        if active_masks is not None:
            dist_entropy = (action_logits.entropy().sum(1)*active_masks.squeeze(-1)).sum()/active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy
    
class ACTLayer_old(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
        super(ACTLayer_old, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        
        # print(action_space,'actionspace')
        # print("Box")
        action_dim = 5
        self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
            
    def forward(self, x, available_actions=None, deterministic=False):
        action_logits = self.action_out(x)
        actions = action_logits.mode() if deterministic else action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)
        
        return actions, action_log_probs

    def get_probs(self, x, available_actions=None):
        action_logits = self.action_out(x, available_actions)
        action_probs = action_logits.probs
        
        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        action_logits = self.action_out(x)
        action_log_probs = action_logits.log_probs(action)

        if active_masks is not None:
            dist_entropy = (action_logits.entropy().sum(1)*active_masks.squeeze(-1)).sum()/active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy