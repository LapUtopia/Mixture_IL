__author__ = "Jie Ren"
__email__ = "jieren9806@gmail.com"

import collections
import random
import torch


class ReplayBuffer:
    def __init__(self, maxlen, device):
        self.buffer = collections.deque(maxlen=maxlen)
        self.device = device

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        s_lst = torch.tensor(s_lst, dtype=torch.float, device=self.device)
        a_lst = torch.tensor(a_lst, device=self.device)
        r_lst = torch.tensor(r_lst, device=self.device)
        s_prime_lst = torch.tensor(s_prime_lst, dtype=torch.float, device=self.device)
        done_mask_lst = torch.tensor(done_mask_lst, device=self.device)

        return s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst

    def size(self):
        return len(self.buffer)


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta = torch.tensor(0.1, device=mu.device)
        self.dt = torch.tensor(0.01, device=mu.device)
        self.sigma = torch.tensor(0.1, device=mu.device)
        self.mu = mu
        self.x_prev = torch.zeros_like(mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * torch.sqrt(self.dt) * torch.randn_like(self.mu)
        self.x_prev = x
        return x
