import numpy as np
import random
import torch

from collections import deque
from gymnasium.spaces import MultiDiscrete

from backtothecode_gym.envs import BackToTheCodeEnvParams
from backtothecode_gym.envs.lib import utils
from backtothecode_players.ai import AIPlayer

from .lib import Linear_QNet, QTrainer

MAX_MEMORY = 10_000
BATCH_SIZE = 32
LR = 0.002

class MultiFeatureAIPlayer(AIPlayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = True
        self.num_batches_learnt = 0
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.build_model()
        
    def _get_random_move_prob(self):
        return 0.2 + 0.8*0.98**self.num_batches_learnt

    def move(self, board, verbose=False):
        # random moves: tradeoff exploration / exploitation
        if self.is_training and np.random.rand() < self._get_random_move_prob():
            possible_actions = board.get_possible_actions(self.id)
            return np.random.choice(list(possible_actions))
        obs = self.observe(board)
        prediction = self.run_model(obs)
        if verbose:
            print(obs, "->", prediction)
        return torch.argmax(prediction).item()

    # Overwrite the function below in derived classes.

    def get_observation_space(self):
        return MultiDiscrete([
            3, 3, 3, 3, # distance to empty cell in all directions
            2, 2, 2, 2 # where do I come from
        ])
    
    def _get_observation_space_size(self):
        return 8
    
    
    def build_model(self):
        self.model = Linear_QNet(
            [self._get_observation_space_size(), 
             32,
             BackToTheCodeEnvParams.NUM_ACTIONS]
        )
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def observe(self, board):
        position = board.get_position(self.id)
        neighbor_positions = []
        distances = []
        for direction in utils.get_directions():
            neighbor_positions.append(utils.move_in_direction(position, direction))
            distances.append(100)
        for cell in board.get_free_cells():
            for i, nb in enumerate(neighbor_positions):
                if board.within_board(nb):
                    d = utils.calculate_distance(nb, cell)
                    distances[i] = min(distances[i], d)
        last_position = board.get_last_position(self.id)
        last_positions = [0] * 4
        # print(last_position, position)
        last_rev_action = utils.get_action_from_neighboring_positions(position, last_position)
        if last_rev_action != -1:
            last_positions[last_rev_action] = 1
        return np.array(distances + last_positions).flatten()