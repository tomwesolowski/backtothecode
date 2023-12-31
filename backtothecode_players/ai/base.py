import numpy as np
import random
import torch

from collections import deque
from gymnasium.spaces import MultiDiscrete
from tqdm import tqdm

from backtothecode_gym.envs import BackToTheCodeEnvParams
from backtothecode_gym.envs.lib import utils
from backtothecode_players.player import Player

from .lib import Linear_QNet, QTrainer

MAX_MEMORY = 10_000
BATCH_SIZE = 32
LR = 0.002

class AIPlayer(Player):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = True
        self.num_batches_learnt = 0
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.build_model()

        
    def reset(self, is_training=False):
        super().reset(is_training)
        self.rewards = []
        
    
    def move(self, board, verbose=False):
        # random moves: tradeoff exploration / exploitation
        if self.is_training and np.random.rand() < 0.99*self.num_batches_learnt:
            possible_actions = board.get_possible_actions(self.id)
            return np.random.choice(list(possible_actions))
        prediction = self.run_model(self.observe(board))
        if verbose:
            print(prediction)
        return torch.argmax(prediction).item()
    
    def run_model(self, observations):
        state = torch.tensor(observations, dtype=torch.float)
        return self.model(state)
    
        
    def feedback(self, old_board, action, reward, new_board, done):
        self.rewards.append(reward)
        state = self.observe(old_board)
        next_state = self.observe(new_board)
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train(self):
        shuffled_memory = list(self.memory)
        random.shuffle(shuffled_memory)
        num_batches = len(shuffled_memory) // BATCH_SIZE
        for step in range(num_batches):
            batch = shuffled_memory[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
            states, actions, rewards, next_states, dones = zip(*batch)
            self.trainer.train_step(states, actions, rewards, next_states, dones)
            self.num_batches_learnt += 1
        self.memory.clear()
        
    def save(self, suffix):
        self.trainer.save(self.name, suffix)
    
    def load(self, suffix):
        self.trainer.load(self.name, suffix)

    # Overwrite the function below in derived classes.

    def get_observation_space(self):
        return MultiDiscrete([
            3, 3, 3, 3 # 4 cells around me
        ])
    
    def _get_observation_space_size(self):
        return 4
    
    
    def build_model(self):
        self.model = Linear_QNet(
            [self._get_observation_space_size(), 
             64, 
             BackToTheCodeEnvParams.NUM_ACTIONS]
        )
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def observe(self, board):
        neighbourhood = []
        position = board.get_position(self.id)
        for direction in utils.get_directions():
            neighbor_position = utils.move_in_direction(position, direction)
            if board.within_board(neighbor_position):
                owner = board.get_ownership(neighbor_position)
                if owner == -1:
                    neighbourhood.append(1) # free cell
                else:
                    neighbourhood.append(0) # owned by someone
            else:
                neighbourhood.append(-1) # forbidden
        return np.array(neighbourhood).flatten()