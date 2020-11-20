import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import ppo.net as net


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return np.array(
        scipy.signal.lfilter([1], [1, float(-discount)], x[:, ::-1], axis=1)[:, ::-1]
    )


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class CategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, arch, **kwargs):
        super().__init__()
        self.logits_net = net.__dict__[arch](obs_dim, act_dim, **kwargs)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class Critic(nn.Module):
    def __init__(self, obs_dim, arch, **kwargs):
        super().__init__()
        self.v_net = net.__dict__[arch](obs_dim, 1, **kwargs)

    def forward(self, obs):
        return torch.squeeze(
            self.v_net(obs), -1
        )  # Critical to ensure v has right shape.


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, arch, **kwargs):
        super().__init__()

        # policy builder depends on action space
        self.pi = CategoricalActor(obs_dim, act_dim, arch, **kwargs)

        # build value function
        self.v = Critic(obs_dim, arch, **kwargs)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]
