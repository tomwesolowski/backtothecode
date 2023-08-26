import numpy as np

from .player import Player

class RandomPlayer(Player):
    def __init__(self, id, name, momentum=0):
        super().__init__(id, name)
        self.momentum = momentum
        self.last_action = None
    
    def move(self, board):
        possible_actions = list(board.get_possible_actions(self.id))
        if self.last_action in possible_actions and np.random.rand() < self.momentum:
            return self.last_action
        self.last_action = np.random.choice(possible_actions)
        return self.last_action