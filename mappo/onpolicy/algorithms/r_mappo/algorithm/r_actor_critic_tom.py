import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.attention import AttentionBase
from onpolicy.algorithms.utils.embed import Embed_sheep_wolf, Embed_landmark, Embed_sheep_wolf_landmark
from onpolicy.algorithms.utils.tom import TOM, TOM_Attention, TOM_Initialization, TOM_Attention_Rearrange, TOM_Attention_Rearrange_Duplicate, TOM_MLP_Value, TOM_ATTENTION_Value
from onpolicy.algorithms.utils.inversenet import InverseNet, InverseNet_Landmark, InverseNet_sheep_wolf_landmark
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.newact import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space

class R_Actor_TOM(nn.Module):
    def __init__(self, args, obs_space, action_space, agent_num, train_module, agent_type, device=torch.device("cpu"), seed = 1):
        super(R_Actor_TOM, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.obs_shape = get_shape_from_obs_space(obs_space)[0]
        self.num_gf = int(self.obs_shape/2)
        self.agent_num = agent_num
        self.train_module = train_module
        self.agent_type = agent_type
        self.num_wolf = args.num_adv
        self.num_sheep = args.num_agents - args.num_adv
        self.num_agent = args.num_agents
        self.num_landmarks = args.num_landmarks
        self.scenario = args.scenario_name

        if self.scenario == "simple_lineup_onlyfood_withoutcredit_humanshape":
            self.embed = Embed_sheep_wolf(input_dim = 2, hidden_dim = 32, agent_num = self.agent_num, train_module = self.train_module, seed = seed)
        elif self.scenario == "simple_landmark":
            self.embed = Embed_landmark(input_dim = 2, hidden_dim = 32, agent_num = self.agent_num, train_module = self.train_module, seed = seed)
        elif self.scenario == "simple_sheep_wolf_landmark":
            self.embed = Embed_sheep_wolf_landmark(input_dim = 2, hidden_dim = 32, agent_num = self.agent_num, train_module = self.train_module, seed = seed)
        
        self.base = AttentionBase(dk = 32, hidden_dim = 32, agent_num = self.agent_num, train_module = self.train_module, seed = seed)
        
        self.act = ACTLayer(action_space, inputs_dim = 32, use_orthogonal = self._use_orthogonal, gain = self._gain, hidden_dim = 64, 
                            agent_num = self.agent_num, train_module = self.train_module, seed = seed)
        
        # self.imagined_we = TOM(input_dim = (num_agent+1)*3, hidden_dim = 64, output_dim = (num_agent+1))
        # self.imagined_we = TOM_Attention(input_dim = num_agent+1, dk = 32)
        self.imagined_we = TOM_Initialization(input_dim = (self.num_agent+1+self.num_landmarks)*3, output_dim = (self.num_agent+1+self.num_landmarks))
        # self.imagined_we = TOM_Attention_Rearrange(input_dim = num_agent+1, dk = 32, agent_num = self.agent_num)
        # self.imagined_we = TOM_Attention_Rearrange_Duplicate(input_dim = num_agent+1, dk = 32, agent_num = self.agent_num, n = 10)
        # self.imagined_we = TOM_MLP_Value(input_dim = (num_agent+1)*3, hidden_dim = 64, output_dim = (num_agent+1))
        # self.imagined_we = TOM_ATTENTION_Value(input_dim = num_agent+1, dk = 32, agent_num = self.agent_num)
        
        if self.scenario == "simple_lineup_onlyfood_withoutcredit_humanshape":
            self.inverse_net = InverseNet(input_dim = 32, hidden_dim = 32, dk = 32, agent_num = self.agent_num, seed = seed) 
        elif self.scenario == "simple_landmark":
            self.inverse_net = InverseNet_Landmark(input_dim = 32, hidden_dim = 32, dk = 32, agent_num = self.agent_num, seed = seed)
        elif self.scenario == "simple_sheep_wolf_landmark":
            self.inverse_net = InverseNet_sheep_wolf_landmark(input_dim = 32, hidden_dim = 32, dk = 32, agent_num = self.agent_num, seed = seed)
        
        self.to(device)

        # print(f"Creating TOM Actor Class: Agent{agent_num}, {agent_type}, load network is {load}, train module is {train_module}")

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        obs = c

        if self.scenario == "simple_lineup_onlyfood_withoutcredit_humanshape":
            if self.num_agent == 4:
                my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf, ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf = self.transform_obs(obs)
                
                my_velocity_embed, my_wolf_gf_embed, my_sheep_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf)
                _, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_wall_gf_embed = self.embed(ally1_wolf_gf, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_wolf_gf_embed, my_sheep_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_wall_gf_embed)
                  
                all_weight = torch.cat((my_weight, ally1_weight), dim = 1)
            
            elif self.num_agent == 6:
                my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf, ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf, ally2_action, ally2_wolf_gf, ally2_sheep_gf, ally2_wall_gf = self.transform_obs(obs)
                
                my_velocity_embed, my_wolf_gf_embed, my_sheep_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf)
                _, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_wall_gf_embed = self.embed(ally1_wolf_gf, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf)
                _, ally2_wolf_gf_embed, ally2_sheep_gf_embed, ally2_wall_gf_embed = self.embed(ally2_wolf_gf, ally2_wolf_gf, ally2_sheep_gf, ally2_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_wolf_gf_embed, my_sheep_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_wall_gf_embed)
                ally2_weight, _, ally2_value = self.inverse_net(ally2_action, ally2_wolf_gf_embed, ally2_sheep_gf_embed, ally2_wall_gf_embed)
                all_weight = torch.cat((my_weight, ally1_weight, ally2_weight), dim = 1) # (batch_size, num_weight*3) -> (batch_size, num_weight)
            
            elif self.num_agent == 8:
                my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf, ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf, ally2_action, ally2_wolf_gf, ally2_sheep_gf, ally2_wall_gf, ally3_action, ally3_wolf_gf, ally3_sheep_gf, ally3_wall_gf = self.transform_obs(obs)
                
                my_velocity_embed, my_wolf_gf_embed, my_sheep_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf)
                _, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_wall_gf_embed = self.embed(ally1_wolf_gf, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf)
                _, ally2_wolf_gf_embed, ally2_sheep_gf_embed, ally2_wall_gf_embed = self.embed(ally2_wolf_gf, ally2_wolf_gf, ally2_sheep_gf, ally2_wall_gf)
                _, ally3_wolf_gf_embed, ally3_sheep_gf_embed, ally3_wall_gf_embed = self.embed(ally3_wolf_gf, ally3_wolf_gf, ally3_sheep_gf, ally3_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_wolf_gf_embed, my_sheep_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_wall_gf_embed)
                ally2_weight, _, ally2_value = self.inverse_net(ally2_action, ally2_wolf_gf_embed, ally2_sheep_gf_embed, ally2_wall_gf_embed)
                ally3_weight, _, ally3_value = self.inverse_net(ally3_action, ally3_wolf_gf_embed, ally3_sheep_gf_embed, ally3_wall_gf_embed)
                all_weight = torch.cat((my_weight, ally1_weight, ally2_weight, ally3_weight), dim = 1)
        
        elif self.scenario == "simple_landmark":
            if self.num_agent == 2:
                my_velocity, my_agent_gf, my_landmark_gf, my_wall_gf, ally1_action, ally1_agent_gf, ally1_landmark_gf, ally1_wall_gf = self.transform_obs(obs)

                my_velocity_embed, my_agent_gf_embed, my_landmark_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_agent_gf, my_landmark_gf, my_wall_gf)
                _, ally1_agent_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed = self.embed(ally1_agent_gf, ally1_agent_gf, ally1_landmark_gf, ally1_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_agent_gf_embed, my_landmark_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_agent_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed)
                all_weight = torch.cat((my_weight, ally1_weight), dim = 1)
            elif self.num_agent == 3:
                my_velocity, my_agent_gf, my_landmark_gf, my_wall_gf, ally1_action, ally1_agent_gf, ally1_landmark_gf, ally1_wall_gf, ally2_action, ally2_agent_gf, ally2_landmark_gf, ally2_wall_gf = self.transform_obs(obs)

                my_velocity_embed, my_agent_gf_embed, my_landmark_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_agent_gf, my_landmark_gf, my_wall_gf)
                _, ally1_agent_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed = self.embed(ally1_agent_gf, ally1_agent_gf, ally1_landmark_gf, ally1_wall_gf)
                _, ally2_agent_gf_embed, ally2_landmark_gf_embed, ally2_wall_gf_embed = self.embed(ally2_agent_gf, ally2_agent_gf, ally2_landmark_gf, ally2_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_agent_gf_embed, my_landmark_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_agent_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed)
                ally2_weight, _, ally2_value = self.inverse_net(ally2_action, ally2_agent_gf_embed, ally2_landmark_gf_embed, ally2_wall_gf_embed)
                all_weight = torch.cat((my_weight, ally1_weight, ally2_weight), dim = 1)
            elif self.num_agent == 4:
                my_velocity, my_agent_gf, my_landmark_gf, my_wall_gf, ally1_action, ally1_agent_gf, ally1_landmark_gf, ally1_wall_gf, ally2_action, ally2_agent_gf, ally2_landmark_gf, ally2_wall_gf, ally3_action, ally3_agent_gf, ally3_landmark_gf, ally3_wall_gf = self.transform_obs(obs)

                my_velocity_embed, my_agent_gf_embed, my_landmark_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_agent_gf, my_landmark_gf, my_wall_gf)
                _, ally1_agent_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed = self.embed(ally1_agent_gf, ally1_agent_gf, ally1_landmark_gf, ally1_wall_gf)
                _, ally2_agent_gf_embed, ally2_landmark_gf_embed, ally2_wall_gf_embed = self.embed(ally2_agent_gf, ally2_agent_gf, ally2_landmark_gf, ally2_wall_gf)
                _, ally3_agent_gf_embed, ally3_landmark_gf_embed, ally3_wall_gf_embed = self.embed(ally3_agent_gf, ally3_agent_gf, ally3_landmark_gf, ally3_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_agent_gf_embed, my_landmark_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_agent_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed)
                ally2_weight, _, ally2_value = self.inverse_net(ally2_action, ally2_agent_gf_embed, ally2_landmark_gf_embed, ally2_wall_gf_embed)
                ally3_weight, _, ally3_value = self.inverse_net(ally3_action, ally3_agent_gf_embed, ally3_landmark_gf_embed, ally3_wall_gf_embed)
                all_weight = torch.cat((my_weight, ally1_weight, ally2_weight, ally3_weight), dim = 1)
        
        elif self.scenario == "simple_sheep_wolf_landmark":
            if self.num_agent == 4:
                my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf, ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf = self.transform_obs(obs)

                my_velocity_embed, my_wolf_gf_embed, my_sheep_gf_embed, my_landmark_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf)
                _, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed = self.embed(ally1_wolf_gf, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_wolf_gf_embed, my_sheep_gf_embed, my_landmark_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed)
                all_weight = torch.cat((my_weight, ally1_weight), dim = 1)
            
            elif self.num_agent == 6:
                my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf, ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf, ally2_action, ally2_wolf_gf, ally2_sheep_gf, ally2_landmark_gf, ally2_wall_gf = self.transform_obs(obs)

                my_velocity_embed, my_wolf_gf_embed, my_sheep_gf_embed, my_landmark_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf)
                _, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed = self.embed(ally1_wolf_gf, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf)
                _, ally2_wolf_gf_embed, ally2_sheep_gf_embed, ally2_landmark_gf_embed, ally2_wall_gf_embed = self.embed(ally2_wolf_gf, ally2_wolf_gf, ally2_sheep_gf, ally2_landmark_gf, ally2_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_wolf_gf_embed, my_sheep_gf_embed, my_landmark_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed)
                ally2_weight, _, ally2_value = self.inverse_net(ally2_action, ally2_wolf_gf_embed, ally2_sheep_gf_embed, ally2_landmark_gf_embed, ally2_wall_gf_embed)
                all_weight = torch.cat((my_weight, ally1_weight, ally2_weight), dim = 1)
            
            elif self.num_agent == 8:
                my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf, ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf, ally2_action, ally2_wolf_gf, ally2_sheep_gf, ally2_landmark_gf, ally2_wall_gf, ally3_action, ally3_wolf_gf, ally3_sheep_gf, ally3_landmark_gf, ally3_wall_gf = self.transform_obs(obs)

                my_velocity_embed, my_wolf_gf_embed, my_sheep_gf_embed, my_landmark_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf)
                _, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed = self.embed(ally1_wolf_gf, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf)
                _, ally2_wolf_gf_embed, ally2_sheep_gf_embed, ally2_landmark_gf_embed, ally2_wall_gf_embed = self.embed(ally2_wolf_gf, ally2_wolf_gf, ally2_sheep_gf, ally2_landmark_gf, ally2_wall_gf)
                _, ally3_wolf_gf_embed, ally3_sheep_gf_embed, ally3_landmark_gf_embed, ally3_wall_gf_embed = self.embed(ally3_wolf_gf, ally3_wolf_gf, ally3_sheep_gf, ally3_landmark_gf, ally3_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_wolf_gf_embed, my_sheep_gf_embed, my_landmark_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed)
                ally2_weight, _, ally2_value = self.inverse_net(ally2_action, ally2_wolf_gf_embed, ally2_sheep_gf_embed, ally2_landmark_gf_embed, ally2_wall_gf_embed)
                ally3_weight, _, ally3_value = self.inverse_net(ally3_action, ally3_wolf_gf_embed, ally3_sheep_gf_embed, ally3_landmark_gf_embed, ally3_wall_gf_embed)
                all_weight = torch.cat((my_weight, ally1_weight, ally2_weight, ally3_weight), dim = 1)

        weight = self.imagined_we(all_weight).unsqueeze(-1)
        weighted_gf = torch.sum(weight * my_value, dim = 1)

        actions, action_log_probs = self.act(weighted_gf, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        obs = obs.view(obs.size()[0], int(self.obs_shape/2), 2)
        
        if self.scenario == "simple_lineup_onlyfood_withoutcredit_humanshape":
            if self.num_agent == 4:
                my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf, ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf = self.transform_obs(obs)
                
                my_velocity_embed, my_wolf_gf_embed, my_sheep_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf)
                _, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_wall_gf_embed = self.embed(ally1_wolf_gf, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_wolf_gf_embed, my_sheep_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_wall_gf_embed)
                  
                all_weight = torch.cat((my_weight, ally1_weight), dim = 1)
            
            elif self.num_agent == 6:
                my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf, ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf, ally2_action, ally2_wolf_gf, ally2_sheep_gf, ally2_wall_gf = self.transform_obs(obs)
                
                my_velocity_embed, my_wolf_gf_embed, my_sheep_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf)
                _, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_wall_gf_embed = self.embed(ally1_wolf_gf, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf)
                _, ally2_wolf_gf_embed, ally2_sheep_gf_embed, ally2_wall_gf_embed = self.embed(ally2_wolf_gf, ally2_wolf_gf, ally2_sheep_gf, ally2_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_wolf_gf_embed, my_sheep_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_wall_gf_embed)
                ally2_weight, _, ally2_value = self.inverse_net(ally2_action, ally2_wolf_gf_embed, ally2_sheep_gf_embed, ally2_wall_gf_embed)
                all_weight = torch.cat((my_weight, ally1_weight, ally2_weight), dim = 1)
            
            elif self.num_agent == 8:
                my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf, ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf, ally2_action, ally2_wolf_gf, ally2_sheep_gf, ally2_wall_gf, ally3_action, ally3_wolf_gf, ally3_sheep_gf, ally3_wall_gf = self.transform_obs(obs)
                
                my_velocity_embed, my_wolf_gf_embed, my_sheep_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf)
                _, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_wall_gf_embed = self.embed(ally1_wolf_gf, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf)
                _, ally2_wolf_gf_embed, ally2_sheep_gf_embed, ally2_wall_gf_embed = self.embed(ally2_wolf_gf, ally2_wolf_gf, ally2_sheep_gf, ally2_wall_gf)
                _, ally3_wolf_gf_embed, ally3_sheep_gf_embed, ally3_wall_gf_embed = self.embed(ally3_wolf_gf, ally3_wolf_gf, ally3_sheep_gf, ally3_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_wolf_gf_embed, my_sheep_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_wall_gf_embed)
                ally2_weight, _, ally2_value = self.inverse_net(ally2_action, ally2_wolf_gf_embed, ally2_sheep_gf_embed, ally2_wall_gf_embed)
                ally3_weight, _, ally3_value = self.inverse_net(ally3_action, ally3_wolf_gf_embed, ally3_sheep_gf_embed, ally3_wall_gf_embed)
                all_weight = torch.cat((my_weight, ally1_weight, ally2_weight, ally3_weight), dim = 1)
        
        elif self.scenario == "simple_landmark":
            if self.num_agent == 2:
                my_velocity, my_agent_gf, my_landmark_gf, my_wall_gf, ally1_action, ally1_agent_gf, ally1_landmark_gf, ally1_wall_gf = self.transform_obs(obs)

                my_velocity_embed, my_agent_gf_embed, my_landmark_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_agent_gf, my_landmark_gf, my_wall_gf)
                _, ally1_agent_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed = self.embed(ally1_agent_gf, ally1_agent_gf, ally1_landmark_gf, ally1_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_agent_gf_embed, my_landmark_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_agent_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed)
                all_weight = torch.cat((my_weight, ally1_weight), dim = 1)
            elif self.num_agent == 3:
                my_velocity, my_agent_gf, my_landmark_gf, my_wall_gf, ally1_action, ally1_agent_gf, ally1_landmark_gf, ally1_wall_gf, ally2_action, ally2_agent_gf, ally2_landmark_gf, ally2_wall_gf = self.transform_obs(obs)

                my_velocity_embed, my_agent_gf_embed, my_landmark_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_agent_gf, my_landmark_gf, my_wall_gf)
                _, ally1_agent_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed = self.embed(ally1_agent_gf, ally1_agent_gf, ally1_landmark_gf, ally1_wall_gf)
                _, ally2_agent_gf_embed, ally2_landmark_gf_embed, ally2_wall_gf_embed = self.embed(ally2_agent_gf, ally2_agent_gf, ally2_landmark_gf, ally2_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_agent_gf_embed, my_landmark_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_agent_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed)
                ally2_weight, _, ally2_value = self.inverse_net(ally2_action, ally2_agent_gf_embed, ally2_landmark_gf_embed, ally2_wall_gf_embed)
                all_weight = torch.cat((my_weight, ally1_weight, ally2_weight), dim = 1)
            
            elif self.num_agent == 4:
                my_velocity, my_agent_gf, my_landmark_gf, my_wall_gf, ally1_action, ally1_agent_gf, ally1_landmark_gf, ally1_wall_gf, ally2_action, ally2_agent_gf, ally2_landmark_gf, ally2_wall_gf, ally3_action, ally3_agent_gf, ally3_landmark_gf, ally3_wall_gf = self.transform_obs(obs)

                my_velocity_embed, my_agent_gf_embed, my_landmark_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_agent_gf, my_landmark_gf, my_wall_gf)
                _, ally1_agent_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed = self.embed(ally1_agent_gf, ally1_agent_gf, ally1_landmark_gf, ally1_wall_gf)
                _, ally2_agent_gf_embed, ally2_landmark_gf_embed, ally2_wall_gf_embed = self.embed(ally2_agent_gf, ally2_agent_gf, ally2_landmark_gf, ally2_wall_gf)
                _, ally3_agent_gf_embed, ally3_landmark_gf_embed, ally3_wall_gf_embed = self.embed(ally3_agent_gf, ally3_agent_gf, ally3_landmark_gf, ally3_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_agent_gf_embed, my_landmark_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_agent_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed)
                ally2_weight, _, ally2_value = self.inverse_net(ally2_action, ally2_agent_gf_embed, ally2_landmark_gf_embed, ally2_wall_gf_embed)
                ally3_weight, _, ally3_value = self.inverse_net(ally3_action, ally3_agent_gf_embed, ally3_landmark_gf_embed, ally3_wall_gf_embed)
                all_weight = torch.cat((my_weight, ally1_weight, ally2_weight, ally3_weight), dim = 1)
        
        elif self.scenario == "simple_sheep_wolf_landmark":
            if self.num_agent == 4:
                my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf, ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf = self.transform_obs(obs)

                my_velocity_embed, my_wolf_gf_embed, my_sheep_gf_embed, my_landmark_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf)
                _, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed = self.embed(ally1_wolf_gf, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_wolf_gf_embed, my_sheep_gf_embed, my_landmark_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed)
                all_weight = torch.cat((my_weight, ally1_weight), dim = 1)
                
            elif self.num_agent == 6:
                my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf, ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf, ally2_action, ally2_wolf_gf, ally2_sheep_gf, ally2_landmark_gf, ally2_wall_gf = self.transform_obs(obs)

                my_velocity_embed, my_wolf_gf_embed, my_sheep_gf_embed, my_landmark_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf)
                _, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed = self.embed(ally1_wolf_gf, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf)
                _, ally2_wolf_gf_embed, ally2_sheep_gf_embed, ally2_landmark_gf_embed, ally2_wall_gf_embed = self.embed(ally2_wolf_gf, ally2_wolf_gf, ally2_sheep_gf, ally2_landmark_gf, ally2_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_wolf_gf_embed, my_sheep_gf_embed, my_landmark_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed)
                ally2_weight, _, ally2_value = self.inverse_net(ally2_action, ally2_wolf_gf_embed, ally2_sheep_gf_embed, ally2_landmark_gf_embed, ally2_wall_gf_embed)
                all_weight = torch.cat((my_weight, ally1_weight, ally2_weight), dim = 1)
            
            elif self.num_agent == 8:
                my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf, ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf, ally2_action, ally2_wolf_gf, ally2_sheep_gf, ally2_landmark_gf, ally2_wall_gf, ally3_action, ally3_wolf_gf, ally3_sheep_gf, ally3_landmark_gf, ally3_wall_gf = self.transform_obs(obs)

                my_velocity_embed, my_wolf_gf_embed, my_sheep_gf_embed, my_landmark_gf_embed, my_wall_gf_embed = self.embed(my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf)
                _, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed = self.embed(ally1_wolf_gf, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf)
                _, ally2_wolf_gf_embed, ally2_sheep_gf_embed, ally2_landmark_gf_embed, ally2_wall_gf_embed = self.embed(ally2_wolf_gf, ally2_wolf_gf, ally2_sheep_gf, ally2_landmark_gf, ally2_wall_gf)
                _, ally3_wolf_gf_embed, ally3_sheep_gf_embed, ally3_landmark_gf_embed, ally3_wall_gf_embed = self.embed(ally3_wolf_gf, ally3_wolf_gf, ally3_sheep_gf, ally3_landmark_gf, ally3_wall_gf)

                my_obs_embed = torch.cat((torch.unsqueeze(my_velocity_embed, 1), my_wolf_gf_embed, my_sheep_gf_embed, my_landmark_gf_embed, torch.unsqueeze(my_wall_gf_embed, 1)), dim = 1)
                my_weight, _, my_value = self.base(my_obs_embed)

                ally1_weight, _, ally1_value = self.inverse_net(ally1_action, ally1_wolf_gf_embed, ally1_sheep_gf_embed, ally1_landmark_gf_embed, ally1_wall_gf_embed)
                ally2_weight, _, ally2_value = self.inverse_net(ally2_action, ally2_wolf_gf_embed, ally2_sheep_gf_embed, ally2_landmark_gf_embed, ally2_wall_gf_embed)
                ally3_weight, _, ally3_value = self.inverse_net(ally3_action, ally3_wolf_gf_embed, ally3_sheep_gf_embed, ally3_landmark_gf_embed, ally3_wall_gf_embed)
                all_weight = torch.cat((my_weight, ally1_weight, ally2_weight, ally3_weight), dim = 1)

        weight = self.imagined_we(all_weight).unsqueeze(-1)
        weighted_gf = torch.sum(weight * my_value, dim = 1)


        action_log_probs, dist_entropy = self.act.evaluate_actions(weighted_gf,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy
    
    def transform_obs(self, obs):
        if self.scenario == "simple_lineup_onlyfood_withoutcredit_humanshape":
            if self.agent_type == "wolf":
                my_velocity = obs[:,0,:]
                my_wolf_gf = obs[:,1:1+(self.num_wolf-1),:]
                my_sheep_gf = obs[:,1+(self.num_wolf-1):1+(self.num_wolf-1)+self.num_sheep,:]
                my_wall_gf = obs[:,1+(self.num_wolf-1)+self.num_sheep,:]
                if self.num_agent >= 4:
                    ally1_action = torch.cat(tuple(obs[:, i, :] for i in range(1+(self.num_wolf-1)+self.num_sheep+1, 1+(self.num_wolf-1)+self.num_sheep+4)), dim=1)[:, :-1]
                    ally1_wolf_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+4:1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1), :]
                    ally1_sheep_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1):1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep, :]
                    ally1_wall_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep, :]

                    if self.num_agent >= 6:
                        ally2_action = torch.cat(tuple(obs[:, i, :] for i in range(1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+1, 1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4)), dim=1)[:, :-1]
                        ally2_wolf_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4:1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1), :]
                        ally2_sheep_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1):1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep, :]
                        ally2_wall_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep, :]

                        if self.num_agent == 8:
                            ally3_action = torch.cat(tuple(obs[:, i, :] for i in range(1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+1, 1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4)), dim=1)[:, :-1]
                            ally3_wolf_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4 : 1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1), :]
                            ally3_sheep_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1) : 1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep, :]
                            ally3_wall_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep+4+(self.num_wolf-1)+self.num_sheep, :]

                            return my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf, \
                                ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf, \
                                ally2_action,ally2_wolf_gf, ally2_sheep_gf, ally2_wall_gf, \
                                ally3_action,ally3_wolf_gf, ally3_sheep_gf, ally3_wall_gf
                        else:
                            return my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf, \
                                ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf, \
                                ally2_action,ally2_wolf_gf, ally2_sheep_gf, ally2_wall_gf

                    
                    else:
                        return my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf, \
                            ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf

                elif self.agent_type == "sheep":
                    my_velocity = obs[:,0,:]
                    my_wolf_gf = obs[:,1:1+self.num_wolf,:]
                    my_sheep_gf = obs[:,1+self.num_wolf:1+self.num_wolf+(self.num_sheep-1),:]
                    my_wall_gf = obs[:,1+self.num_wolf+(self.num_sheep-1),:]

                    if self.num_agent >= 4:
                        ally1_action = torch.cat(tuple(obs[:, i, :] for i in range(1+self.num_wolf+(self.num_sheep-1)+1, 1+self.num_wolf+(self.num_sheep-1)+4)), dim=1)[:, :-1]
                        ally1_wolf_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+4:1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf, :]
                        ally1_sheep_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf:1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1), :]
                        ally1_wall_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1), :]

                        if self.num_agent >= 6:
                            ally2_action = torch.cat(tuple(obs[:, i, :] for i in range(1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+1, 1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4)), dim=1)[:, :-1]
                            ally2_wolf_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4:1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf, :]
                            ally2_sheep_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf:1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1), :]
                            ally2_wall_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1), :]

                            if self.num_agent == 8:
                                ally3_action = torch.cat(tuple(obs[:, i, :] for i in range(1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+1, 1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4)), dim=1)[:, :-1]
                                ally3_wolf_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4 : 1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf, :]
                                ally3_sheep_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf : 1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1), :]
                                ally3_wall_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1)+4+self.num_wolf+(self.num_sheep-1), :]

                                return my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf, \
                                    ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf, \
                                    ally2_action,ally2_wolf_gf, ally2_sheep_gf, ally2_wall_gf, \
                                    ally3_action,ally3_wolf_gf, ally3_sheep_gf, ally3_wall_gf
                            else:
                                return my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf, \
                                    ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf, \
                                    ally2_action,ally2_wolf_gf, ally2_sheep_gf, ally2_wall_gf

                        
                        else:
                            return my_velocity, my_wolf_gf, my_sheep_gf, my_wall_gf, \
                                ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_wall_gf
            
        elif self.scenario == "simple_landmark":
                my_velocity = obs[:,0,:]
                my_agent_gf = obs[:,1:1+(self.num_agent-1),:]
                my_landmark_gf = obs[:,1+(self.num_agent-1):1+(self.num_agent-1)+self.num_landmarks,:]
                my_wall_gf = obs[:,1+(self.num_agent-1)+self.num_landmarks,:]

                if self.num_agent >= 2:
                    ally1_action = torch.cat(tuple(obs[:, i, :] for i in range(1+(self.num_agent-1)+self.num_landmarks+1, 1+(self.num_agent-1)+self.num_landmarks+4)), dim=1)[:, :-1]
                    ally1_agent_gf = obs[:,1+(self.num_agent-1)+self.num_landmarks+4:1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1),:]
                    ally1_landmark_gf = obs[:,1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1):1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks,:]
                    ally1_wall_gf = obs[:,1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks,:]

                    if self.num_agent >= 3:
                        ally2_action = torch.cat(tuple(obs[:, i, :] for i in range(1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+1 , 1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4 )), dim=1)[:, :-1]
                        ally2_agent_gf = obs[:, 1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4 : 1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1),:]
                        ally2_landmark_gf = obs[:, 1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1) : 1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks,:]
                        ally2_wall_gf = obs[:, 1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks ,:]

                        if self.num_agent == 4:
                            ally3_action = torch.cat(tuple(obs[:, i, :] for i in range(1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+1 , 1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4 )), dim=1)[:, :-1]
                            ally3_agent_gf = obs[:, 1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4 : 1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1),:]
                            ally3_landmark_gf = obs[:, 1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1) : 1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks,:]
                            ally3_wall_gf = obs[:, 1+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks+4+(self.num_agent-1)+self.num_landmarks ,:]

                            return my_velocity, my_agent_gf, my_landmark_gf, my_wall_gf, \
                                ally1_action, ally1_agent_gf, ally1_landmark_gf, ally1_wall_gf, \
                                ally2_action, ally2_agent_gf, ally2_landmark_gf, ally2_wall_gf, \
                                ally3_action, ally3_agent_gf, ally3_landmark_gf, ally3_wall_gf
                        else:
                            return my_velocity, my_agent_gf, my_landmark_gf, my_wall_gf, \
                                ally1_action, ally1_agent_gf, ally1_landmark_gf, ally1_wall_gf, \
                                ally2_action, ally2_agent_gf, ally2_landmark_gf, ally2_wall_gf
                            
                    else:
                        return my_velocity, my_agent_gf, my_landmark_gf, my_wall_gf, \
                            ally1_action, ally1_agent_gf, ally1_landmark_gf, ally1_wall_gf
        
        elif self.scenario == "simple_sheep_wolf_landmark":
                if self.agent_type == "wolf":
                    my_velocity = obs[:,0,:]
                    my_wolf_gf = obs[:,1:1+(self.num_wolf-1),:]
                    my_sheep_gf = obs[:,1+(self.num_wolf-1):1+(self.num_wolf-1)+self.num_sheep,:]
                    my_landmark_gf = obs[:,1+(self.num_wolf-1)+self.num_sheep:1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks,:]
                    my_wall_gf = obs[:,1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks,:]

                    if self.num_agent >= 4:
                        ally1_action = torch.cat(tuple(obs[:, i, :] for i in range(1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+1, 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4)), dim=1)[:, :-1]
                        ally1_wolf_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4:1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1), :]
                        ally1_sheep_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1):1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep, :]
                        ally1_landmark_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep:1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks, :]
                        ally1_wall_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks, :]

                        if self.num_agent >= 6:
                            ally2_action = torch.cat(tuple(obs[:, i, :] for i in range(1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+1, 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4)), dim=1)[:, :-1]
                            ally2_wolf_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4 : 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1), :]
                            ally2_sheep_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1) : 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep, :]
                            ally2_landmark_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep : 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks, :]
                            ally2_wall_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks, :]

                            if self.num_agent == 8:
                                ally3_action = torch.cat(tuple(obs[:, i, :] for i in range(1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+1, 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4)), dim=1)[:, :-1]
                                ally3_wolf_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4 : 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1), :]
                                ally3_sheep_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1) : 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep, :]
                                ally3_landmark_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep : 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks, :]
                                ally3_wall_gf = obs[:, 1+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks+4+(self.num_wolf-1)+self.num_sheep+self.num_landmarks, :]

                                return my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf,\
                                    ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf, \
                                    ally2_action, ally2_wolf_gf, ally2_sheep_gf, ally2_landmark_gf, ally2_wall_gf, \
                                    ally3_action, ally3_wolf_gf, ally3_sheep_gf, ally3_landmark_gf, ally3_wall_gf

                            else:
                                return my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf,\
                                    ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf, \
                                    ally2_action, ally2_wolf_gf, ally2_sheep_gf, ally2_landmark_gf, ally2_wall_gf

                        else:
                            return my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf,\
                                ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf
                    
                elif self.agent_type == "sheep":
                    my_velocity = obs[:,0,:]
                    my_wolf_gf = obs[:,1:1+self.num_wolf,:]
                    my_sheep_gf = obs[:,1+self.num_wolf:1+self.num_wolf+(self.num_sheep-1),:]
                    my_landmark_gf = obs[:,1+self.num_wolf+(self.num_sheep-1):1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks,:]
                    my_wall_gf = obs[:,1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks,:]

                    if self.num_agent >= 4:
                        ally1_action = torch.cat(tuple(obs[:, i, :] for i in range(1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+1, 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4)), dim=1)[:, :-1]
                        ally1_wolf_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4:1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf, :]
                        ally1_sheep_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf:1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1), :]
                        ally1_landmark_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1):1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks, :]
                        ally1_wall_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks, :]

                        if self.num_agent >= 6:
                            ally2_action = torch.cat(tuple(obs[:, i, :] for i in range(1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+1, 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4)), dim=1)[:, :-1]
                            ally2_wolf_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4 : 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf, :]
                            ally2_sheep_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf : 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1), :]
                            ally2_landmark_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1) : 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks, :]
                            ally2_wall_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks, :]

                            if self.num_agent == 8:
                                ally3_action = torch.cat(tuple(obs[:, i, :] for i in range(1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+1, 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4)), dim=1)[:, :-1]
                                ally3_wolf_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4 : 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf, :]
                                ally3_sheep_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf : 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1), :]
                                ally3_landmark_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1) : 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks, :]
                                ally3_wall_gf = obs[:, 1+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks+4+self.num_wolf+(self.num_sheep-1)+self.num_landmarks, :]

                                return my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf,\
                                    ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf, \
                                    ally2_action, ally2_wolf_gf, ally2_sheep_gf, ally2_landmark_gf, ally2_wall_gf, \
                                    ally3_action, ally3_wolf_gf, ally3_sheep_gf, ally3_landmark_gf, ally3_wall_gf

                            else:
                                return my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf,\
                                    ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf, \
                                    ally2_action, ally2_wolf_gf, ally2_sheep_gf, ally2_landmark_gf, ally2_wall_gf

                        else:
                            return my_velocity, my_wolf_gf, my_sheep_gf, my_landmark_gf, my_wall_gf,\
                                ally1_action, ally1_wolf_gf, ally1_sheep_gf, ally1_landmark_gf, ally1_wall_gf
        

class R_Critic_TOM(nn.Module):
    def __init__(self, args, cent_obs_space, agent_num, train_module, device=torch.device("cpu"), seed = 1):
        super(R_Critic_TOM, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)

        cent_obs_shape = tuple((args.num_agents*(args.num_agents+1+args.num_landmarks)*2,))
        # print(cent_obs_shape)
        # print(agent_num)
        base = MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)
        
        if not train_module:
            print(f"AGENT{agent_num} Critic Network Not Train")
            for param in self.parameters():
                param.requires_grad = False

        self.num_agent = args.num_agents
        self.num_landmark = args.num_landmarks

    def forward(self, cent_obs, rnn_states, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)

        slices = []

        start = 0
        end = start + (self.num_agent+self.num_landmark+1)*2
        for i in range(self.num_agent):
            slices.append(cent_obs[:, start:end])
            start = end + (3+self.num_agent+self.num_landmark)*2*2 if i == 1 else end
            end = start + (self.num_agent+self.num_landmark+1)*2
            
        cent_obs = torch.cat(slices, dim=1)

        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states

