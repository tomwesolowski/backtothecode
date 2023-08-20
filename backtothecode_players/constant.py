import numpy as np

from backtothecode_gym.envs.lib import utils

from .player import Player

class ConstantPlayer(Player):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_action = None
    
    def move(self):
        if self.last_action:
            self.last_action = utils.get_opposite_action(self.last_action)
        else:
            possible_actions = list(self.board.get_possible_actions(self.id))
            self.last_action = np.random.choice(possible_actions)
        return self.last_action 