import numpy as np

from .player import Player

class KeyboardPlayer(Player):
    def __init__(self):
        super().__init__()
        self.keys_to_actions = {
            'w': 2,
            'a': 0,
            'd': 1,
            's': 3
        }

    def move(self, board):
        while True:
            direction = input()
            action = self.keys_to_actions.get(direction, -1)
            possible_actions = list(board.get_possible_actions(self.id))
            if action in possible_actions:
                return action
            
        
        return np.random.choice(possible_actions)