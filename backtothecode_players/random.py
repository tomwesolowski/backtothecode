import numpy as np

from .player import Player

class RandomPlayer(Player):
    def move(self):
        possible_actions = list(self.board.get_possible_actions(self.id))
        return np.random.choice(possible_actions)