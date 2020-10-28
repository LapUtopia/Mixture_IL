__author__ = "Jie Ren"
__email__ = "jieren9806@gmail.com"
# ----------------------
# The class tanh_normal and get_action_info is copied from https://github.com/vitchyr/rlkit, thank you vitchyr.
# We use vitchyr's code and modify the code under the MIT License.
# ----------------------
import torch
import random
import numpy as np
from torch import nn
from torch.distributions import Distribution, Normal
import gym
import os


def set_seed(seed, env):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)


def get_activation(name):
    if name == "ReLU":
        return nn.ReLU()
    elif name == "LeakyReLU":
        return nn.LeakyReLU()
    elif name == "Tanh":
        return nn.Tanh()
    elif name == "Sigmoid":
        return nn.Sigmoid()
    elif name is None:
        return None


def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


class tanh_normal(Distribution):
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        mean = torch.zeros_like(self.normal_mean)
        std = torch.ones_like(self.normal_mean)
        z = self.normal_mean + self.normal_std * Normal(mean, std).sample()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)


# get action_infos
class get_action_info:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.dist = tanh_normal(normal_mean=self.mean, normal_std=self.std)

    # select actions
    def select_actions(self, exploration=True, reparameterize=True):
        if exploration:
            if reparameterize:
                actions, pretanh = self.dist.rsample(return_pretanh_value=True)
                return actions, pretanh
            else:
                actions = self.dist.sample()
        else:
            actions = torch.tanh(self.mean)
        return actions

    def get_log_prob(self, actions, pre_tanh_value):
        log_prob = self.dist.log_prob(actions, pre_tanh_value=pre_tanh_value)
        return log_prob.sum(dim=1, keepdim=True)


def set_requires_grad(net, state):
    for p in net.parameters():
        p.requires_grad = state


def get_env_info(env):
    action_space = env.action_space
    observation_space = env.observation_space

    if isinstance(action_space, gym.spaces.Discrete):
        discrete = True
    else:
        discrete = False

    info = {
        "action_space": action_space,
        "observation_space": observation_space,
        "discrete": discrete,
        "action_scale": action_space.high if not discrete else None
    }
    return info


def visualize(window, env_info, agent_index, pi, scaled_action, k):
    import pygame
    RED = 255, 0, 0
    ORANGE = 255, 156, 0
    YELLOW = 255, 255, 0
    GREEN = 0, 255, 0
    GREENBLUE = 0, 255, 255
    BLUE = 0, 0, 255
    PURPLE = 255, 0, 255
    BLACK = 0, 0, 0
    WHITE = 255, 255, 255

    font = pygame.font.Font('times.ttf', 100)
    text = font.render(f'{agent_index}', True, BLACK, WHITE)
    window.blit(text, (0, 0))

    for i in range(k):
        font = pygame.font.Font('times.ttf', 50)
        text = font.render(f'{i}:{pi[i]:.2%}', True, BLACK, WHITE)
        window.blit(text, (0, 100 + i * 50))
        for j in range(env_info["action_space"].shape[0]):
            pygame.draw.rect(window, RED, (350 + 250 * j, 100 + i * 50, scaled_action[i, j] * 100, 40))

    for j in range(env_info["action_space"].shape[0]):
        pygame.draw.line(window, BLACK, (350 + 250 * j, 100), (350 + 250 * j, 100 + 50 * k - 10), 10)
    pygame.display.flip()
    window.fill(WHITE)


def write_logs(writer, logs, epoch):
    for name in logs:
        writer.add_scalar(name, logs[name], epoch)


def recurrent_mkdir(path):
    path = path.split('/')
    sub_path = ''
    for item in path:
        sub_path = os.path.join(sub_path, item)
        if os.path.exists(sub_path):
            pass
        else:
            os.mkdir(sub_path)
