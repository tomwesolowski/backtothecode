import numpy as np

from gymnasium.spaces import Discrete, MultiDiscrete

from backtothecode_gym.envs.lib.board import ReadOnlyBoard

class Player:
    def __init__(self, id, name, board : ReadOnlyBoard):
        self.id = id
        self.name = name
        self.board = board
        self.reset()

    def reset(self):
        self.score = 0

    def get_observation_space(self):
        return MultiDiscrete(
            list(self.board.shape) + # hero's position
            list(self.board.shape) # opponent's position
        )

    def observe(self):
        observations = []
        for player_id in range(self.board.num_players):
            observations.extend(list(self.board.get_position(player_id)))
        return np.array(observations)

    def move(self):
      raise NotImplementedError()