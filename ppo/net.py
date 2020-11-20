import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["mlp", "ds", "rnn"]


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=64, num_layer=2, activation=nn.ReLU):
        super(MLP, self).__init__()
        self.act = activation
        layers = [nn.Linear(dim_in, dim_hidden), self.act()]

        for i in range(num_layer - 1):
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(self.act())

        layers.append(nn.Linear(dim_hidden, dim_out))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        return self.fc(x.view(batch_size, -1))


class DeepSet(nn.Module):
    # Zaheer et al., NIPS (2017). https://arxiv.org/abs/1703.06114
    def __init__(self, dim_in, dim_out, dim_hidden=64, activation=nn.ReLU):
        super(DeepSet, self).__init__()
        self.act = activation
        self.enc = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            self.act(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            self.act(),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x):
        x = self.enc(x).mean(-2)
        x = self.dec(x)
        return x


class RNN(nn.Module):
    def __init__(self, dim_in, dim_out, n_layer=1, dim_hidden=64, activation=nn.ReLU):
        super(RNN, self).__init__()
        self.act = activation
        self.n_layer = n_layer
        self.dim_hidden = dim_hidden
        self.dim_rnn = dim_hidden // 2
        self.emb = nn.Embedding(2, self.dim_rnn // 2)
        self.rnn = nn.GRU(self.dim_rnn // 2, self.dim_rnn, n_layer)
        self.enc = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            self.act(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.dec = nn.Sequential(
            nn.Linear(self.dim_hidden + self.dim_rnn, self.dim_hidden),
            self.act(),
            nn.Linear(self.dim_hidden, dim_out),
        )

    def forward(self, s):
        obs, hist_act = s

        hist_act = self.emb(hist_act)
        os, _ = self.rnn(hist_act)

        hist_act = os[-1]
        obs = self.enc(obs).mean(-2)

        x = torch.cat([obs, hist_act], dim=-1)
        output = self.dec(x)
        return output


def mlp(obs_dim, dim_out, **kwargs):
    model = MLP(np.prod(obs_dim), dim_out, **kwargs)
    return model


def ds(obs_dim, dim_out, **kwargs):
    model = DeepSet(obs_dim[-1], dim_out, **kwargs)
    return model


def rnn(obs_dim, dim_out, **kwargs):
    model = RNN(obs_dim[-1], dim_out, **kwargs)
    return model