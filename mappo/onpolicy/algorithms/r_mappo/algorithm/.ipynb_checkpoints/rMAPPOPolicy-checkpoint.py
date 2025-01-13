import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Critic, R_Actor
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_att import R_Critic_ATTENTION, R_Actor_ATTENTION
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_tom import R_Critic_TOM, R_Actor_TOM
from onpolicy.utils.util import update_linear_schedule

class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu"), agent_num = 0, policy_type = "MAPPO"):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        seed = args.seed
        temp = agent_type = "wolf" if agent_num < args.num_adv else "sheep"
        print(f"{policy_type}: {agent_num} {temp}")
        if policy_type == "MAPPO":
            self.actor = R_Actor(args, self.obs_space, self.act_space)
            self.critic = R_Critic(args, self.share_obs_space, self.device)
        elif policy_type == "GF+ATTENTION":
            self.actor = R_Actor_ATTENTION(args, self.obs_space, self.act_space, agent_num = agent_num, train_module = True,
                                           agent_type = "wolf" if agent_num < args.num_adv else "sheep", device = self.device)
            self.critic = R_Critic_ATTENTION(args, self.share_obs_space, agent_num = agent_num, train = True, device = self.device)
        else:
            self.actor = R_Actor_TOM(args, self.obs_space, self.act_space, agent_num = agent_num, train_module = True,
                                           agent_type = "wolf" if agent_num < args.num_adv else "sheep", device = self.device, seed = seed)
            self.critic = R_Critic_TOM(args, self.share_obs_space, agent_num = agent_num, train_module = True, device = self.device, seed = seed)

        # print(f"Load: /root/mappo_LQ/onpolicy/sheep_wolf_landmark4/ATT4 {seed}/files/actor_agent{agent_num}.pt")
        # checkpoint = torch.load(f'/root/mappo_LQ/onpolicy/sheep_wolf_landmark4/ATT4 {seed}/files/actor_agent{agent_num}.pt')
        # model_state_dict = self.actor.state_dict()
        # adjusted_state_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict}
        # self.actor.load_state_dict(adjusted_state_dict, strict=False)
        # print(f"AGENT{agent_num} LOAD: {adjusted_state_dict.keys()}")

        # critic_state_dict = torch.load(f'/root/mappo_LQ/onpolicy/sheep_wolf_landmark4/ATT4 {seed}/files/critic_agent{agent_num}.pt')
        # self.critic.load_state_dict(critic_state_dict)
        # print(f"AGENT{agent_num} CRITIC NETWORK LAOD: {critic_state_dict.keys()}")
            
        print("--------------------")

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        
    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        # print("obsshape",obs.size())
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
