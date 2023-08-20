import numpy as np
import random
import torch

from collections import deque
from backtothecode_gym.envs.lib import utils
from backtothecode_players.player import Player

from .model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class SimpleAIPlayer(Player):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_observation_space(self):
        return MultiDiscrete(
            3, 3, 3, 3 # 4 cells around me
        )

    def observe(self):
        observations = []
        position = self.board.get_position(self.id)
        for direction in utils.get_directions():
            neighbor_position = utils.move_in_direction(position, direction)
            if self.board.within_board(neighbor_position):
                owner = self.board.get_ownership(neighbor_position)
                if owner == -1:
                    observations.append(1) # free cell
                else:
                    observations.append(0) # owned by someone
            else:
                observations.append(-1)
        return np.array(observations)

    def move(self):
       # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def _remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def _train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def _train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)