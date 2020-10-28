__author__ = "Jie Ren"
__email__ = "jieren9806@gmail.com"
# ----------------------
# Use Soft Actor-Critic as backbone, pdf: https://arxiv.org/abs/1812.05905
# Thanks to the pytorch implement released by vitchyr, url: https://github.com/vitchyr/rlkit.
# We use vitchyr's code and modify the code under the MIT License.
# ----------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import utils
from torch import optim
from torch.distributions import Categorical
import numpy as np


class Critic(nn.Module):
    def __init__(self, action_space, observation_space, discrete, params):
        super().__init__()
        if discrete:
            self.action_shape = action_space.n
        else:
            self.action_shape = action_space.shape[0]
        self.observation_size = observation_space.shape[0]
        self.k = params['k']
        self.use_action = params["critic_use_action"]

        net_struct = copy.deepcopy(params["critic_tail"])
        last_arch, activation = net_struct.pop(0)
        tail = []
        if self.use_action:
            tail.append(nn.Linear(self.action_shape+self.observation_size, last_arch))
        else:
            tail.append(nn.Linear(self.observation_size, last_arch))
        activation = utils.get_activation(activation)
        if activation is not None:
            tail.append(activation)

        for arch, activation_name in net_struct:
            tail.append(nn.Linear(last_arch, arch))
            activation = utils.get_activation(activation_name)
            if activation is not None:
                tail.append(activation)
            last_arch = arch

        self.tail = nn.Sequential(*tail)

    def forward(self, obs, action=None, reshape=False):
        if reshape:
            action = action.reshape([-1, self.action_shape])
            obs = obs.unsqueeze(1).repeat(1, self.k, 1).reshape(-1, self.observation_size)
        inputs = torch.cat([obs, action], dim=1) if action is not None else obs
        output = self.tail(inputs)
        if reshape:
            output = output.reshape(-1, self.k)
        return output


class Actor(nn.Module):
    def __init__(self, action_space, observation_space, discrete, params):
        super().__init__()
        if discrete:
            self.action_shape = action_space.n
        else:
            self.action_shape = action_space.shape[0]
        self.observation_size = observation_space.shape[0]
        self.k = params['k']

        net_struct = copy.deepcopy(params["actor_head"])
        last_arch, activation_name = net_struct.pop(0)
        head = [nn.Linear(self.observation_size, last_arch)]
        activation = utils.get_activation(activation_name)
        if activation is not None:
            head.append(activation)

        for arch, activation_name in net_struct:
            head.append(nn.Linear(last_arch, arch))
            activation = utils.get_activation(activation_name)
            if activation is not None:
                head.append(activation)
            last_arch = arch

        self.head = nn.Sequential(*head)

        self.pi = nn.Linear(last_arch, self.k)
        self.mean = nn.Linear(last_arch, self.k*self.action_shape)
        self.log_std = nn.Linear(last_arch, self.k*self.action_shape)
        # the log_std_min and log_std_max
        self.log_std_min = params["log_std_min"]
        self.log_std_max = params["log_std_max"]

    def forward(self, obs):
        x = self.head(obs)
        pi = self.pi(x)
        pi = torch.softmax(pi, -1)
        mean = self.mean(x)
        log_std = self.log_std(x)
        # clamp the log std
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        mean = mean.reshape(-1, self.k, self.action_shape)
        log_std = log_std.reshape(-1, self.k, self.action_shape)
        # the reparameterization trick
        # return mean and std
        return pi, mean, torch.exp(log_std)


class SAC_Agent:
    def __init__(self, env_info, params, device):
        super(SAC_Agent, self).__init__()
        self.params = params
        self.k = params['k']

        action_space = env_info["action_space"]
        observation_space = env_info["observation_space"]
        discrete = env_info["discrete"]

        self.actor = Actor(action_space, observation_space, discrete, params).to(device)
        self.critic_1 = Critic(action_space, observation_space, discrete, params).to(device)
        self.critic_2 = Critic(action_space, observation_space, discrete, params).to(device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)

        utils.set_requires_grad(self.critic_1_target, False)
        utils.set_requires_grad(self.critic_2_target, False)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=params["lr_actor"])
        self.critic_1_optim = optim.Adam(self.critic_1.parameters(), lr=params["lr_critic"])
        self.critic_2_optim = optim.Adam(self.critic_2.parameters(), lr=params["lr_critic"])
        self.alpha_optim = optim.Adam([self.log_alpha], lr=params["lr_alpha"])

        self.target_entropy = -np.prod(self.actor.action_shape).item()

    def take_action(self, observation):
        pi, mean, std = self.actor(observation)
        agent_index = Categorical(pi).sample()
        mean_chosen = mean[:, agent_index]
        std_chosen = std[:, agent_index]
        action = utils.get_action_info(mean_chosen, std_chosen).select_actions(reparameterize=False)
        return pi, agent_index, action

    def take_eval_action(self, observation):
        pi, mean, std = self.actor(observation)
        agent_index = Categorical(pi).sample()
        mean_chosen = mean[:, agent_index]
        std_chosen = std[:, agent_index]
        action = utils.get_action_info(mean_chosen, std_chosen).select_actions(exploration=False, reparameterize=False)
        return pi, agent_index, action

    def take_visualization_action(self, observation):
        pi, mean, std = self.actor(observation)
        agent_index = Categorical(pi).sample()
        action = utils.get_action_info(mean[0], std[0]).select_actions(exploration=False, reparameterize=False)
        action = action.reshape(self.k, action.shape[-1])
        action_chosen = action[agent_index]
        return pi, agent_index, action, action_chosen

    def save_model(self, path, epoch, total_step):
        def _get_model_state():
            state = {
                "actor": self.actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "critic_1_target": self.critic_1_target.state_dict(),
                "critic_2_target": self.critic_2_target.state_dict(),
            }
            return state

        def _get_optimizer_state():
            state = {
                "actor_optim": self.actor_optim.state_dict(),
                "critic_1_optim": self.critic_1_optim.state_dict(),
                "critic_2_optim": self.critic_2_optim.state_dict()
            }
            return state
        state = {"epoch": epoch,
                 "total_step": total_step}
        model_state = _get_model_state()
        optimizer_state = _get_optimizer_state()

        state.update(model_state)
        state.update(optimizer_state)
        torch.save(state, path)

    def load_model(self, path):
        def _load_model_state(state):
            self.actor.load_state_dict(state["actor"])
            self.critic_1.load_state_dict(state["critic_1"])
            self.critic_2.load_state_dict(state["critic_2"])
            self.critic_1_target.load_state_dict(state["critic_1_target"])
            self.critic_2_target.load_state_dict(state["critic_2_target"])
            
        def _load_optimizer_state(state):
            self.actor_optim.load_state_dict(state["actor_optim"])
            self.critic_1_optim.load_state_dict(state["critic_1_optim"])
            self.critic_2_optim.load_state_dict(state["critic_2_optim"])
        state = torch.load(path)
        _load_model_state(state)
        _load_optimizer_state(state)
        return state["epoch"]#, state["total_step"]
        
    def train_epoch(self, buffer):
        obsevation, action, reward, obsevation_prime, done_mask = buffer.sample(self.params["batch_size"])

        # for debug -------------------------------------------------------
        # obsevation = torch.randn(50, 8).to(self.params["device"])
        # action = torch.randn(50, 2).to(self.params["device"])
        # obsevation_prime = torch.randn(50, 8).to(self.params["device"])
        # done_mask = torch.zeros(50).to(self.params["device"])
        # reward = torch.zeros(50).to(self.params["device"])
        # for debug -------------------------------------------------------

        # mean: batch_size X k X action_shape
        pi, mean, std = self.actor(obsevation)

        # mean: batch_size * k X action_shape
        mean = mean.reshape(-1, mean.shape[-1])
        std = std.reshape(-1, std.shape[-1])

        action_info = utils.get_action_info(mean, std)
        action_pred, pre_tanh_value = action_info.select_actions(reparameterize=True)
        log_prob = action_info.get_log_prob(action_pred, pre_tanh_value)

        action_pred = action_pred.reshape(-1, self.k, action.shape[-1])
        log_prob = log_prob.reshape(-1, self.k)

        agent_index = Categorical(pi).sample().unsqueeze(-1)
        log_prob_chosen = torch.gather(log_prob, 1, agent_index).squeeze()

        alpha_loss = -(self.log_alpha * (log_prob_chosen + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        alpha = self.log_alpha.detach()

        q1 = self.critic_1(obsevation, action)
        q2 = self.critic_2(obsevation, action)

        with torch.no_grad():
            pi_next, mean_next, std_next = self.actor(obsevation_prime)
            agent_index_next = Categorical(pi_next).sample().unsqueeze(-1).unsqueeze(-1)
            agent_index_next = agent_index_next.repeat(1, 1, mean_next.shape[-1])
            mean_chosen_next = torch.gather(mean_next, 1, agent_index_next).reshape(-1, mean_next.shape[-1])
            std_chosen_next = torch.gather(std_next, 1, agent_index_next).reshape(-1, std_next.shape[-1])

            action_info_next = utils.get_action_info(mean_chosen_next, std_chosen_next)
            action_chosen_next, pre_tanh_value_next = action_info_next.select_actions(reparameterize=True)
            log_prob_next = action_info_next.get_log_prob(action_chosen_next, pre_tanh_value_next)
            target_q_value_next = torch.min(self.critic_1_target(obsevation_prime, action_chosen_next),
                                            self.critic_2_target(obsevation_prime, action_chosen_next)) - alpha * log_prob_next
            target_q_value = self.params["reward_scale"] * reward + (1 - done_mask.float()) * self.params["gamma"] * target_q_value_next

        q1_loss = F.mse_loss(q1, target_q_value)
        q2_loss = F.mse_loss(q2, target_q_value)
        q_loss = q1_loss + q2_loss

        self.critic_1_optim.zero_grad()
        self.critic_2_optim.zero_grad()
        q_loss.backward()
        self.critic_1_optim.step()
        self.critic_2_optim.step()

        utils.set_requires_grad(self.critic_1, False)
        utils.set_requires_grad(self.critic_2, False)

        q1_for_pred = self.critic_1(obsevation, action_pred, True)
        q2_for_pred = self.critic_2(obsevation, action_pred, True)

        q_for_pred = torch.min(q1_for_pred, q2_for_pred)
        max_index = torch.argmax(q_for_pred, -1)
        actor_q_loss = -torch.gather(q_for_pred, 1, max_index.unsqueeze(1)).mean()
        log_prob_max = torch.gather(log_prob, 1, max_index.unsqueeze(1))
        actor_entropy_loss = (alpha * log_prob_max).mean()

        max_onehot = F.one_hot(max_index, self.k)
        pi_loss = F.mse_loss(pi, max_onehot.float())

        actor_loss = actor_q_loss + actor_entropy_loss + pi_loss

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        utils.set_requires_grad(self.critic_1, True)
        utils.set_requires_grad(self.critic_2, True)

        return q1_loss.item(), q2_loss.item(), actor_loss.item(), pi_loss.item(), alpha_loss.item()

    def soft_update(self):
        utils.soft_update(self.critic_1, self.critic_1_target, self.params["tau"])
        utils.soft_update(self.critic_2, self.critic_2_target, self.params["tau"])
