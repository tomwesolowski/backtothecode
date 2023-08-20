import numpy as np

from backtothecode_gym.envs.lib import utils

from .player import Player

class ConstantPlayer(Player):
    def __init__(self, action, **kwargs):
        super().__init__(**kwargs)
        self.action = action
    
    def move(self):
        return self.action