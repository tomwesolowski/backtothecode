import numpy as np

from gymnasium.spaces import Discrete, MultiDiscrete

from backtothecode_gym.envs.lib.board import ReadOnlyBoard
from backtothecode_gym.envs import BackToTheCodeEnvParams

class Player:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.trainable = False
        self.is_training = False
        self.reset()

    def reset(self, is_training=False):
        self.score = 0
        self.is_training = is_training

    def get_observation_space(self):
        return MultiDiscrete(
            [BackToTheCodeEnvParams.HEIGHT, BackToTheCodeEnvParams.WIDTH] + # hero's position
            [BackToTheCodeEnvParams.HEIGHT, BackToTheCodeEnvParams.WIDTH] # opponent's position
        )

    def observe(self, board):
        observations = []
        for player_id in range(board.num_players):
            observations.extend(list(board.get_position(player_id)))
        return np.array(observations)

    def move(self, board):
        raise NotImplementedError()
    
    def feedback(self, old_board, action, reward, new_board, done):
        raise NotImplementedError()