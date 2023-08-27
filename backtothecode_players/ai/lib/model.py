# Created by patrickloeber@
# https://github.com/patrickloeber/snake-ai-pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


# (batch, observation_space.size) -> (...) -> (action_space.size)
# it approcimates the Q(s, a) function
class Linear_QNet(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = []
        for before, after in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(nn.Linear(before, after))
            self.layers.append(nn.ReLU())
        self.layers.pop()
        self.linear_relu_stack = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.linear_relu_stack(x)