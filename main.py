import gymnasium as gym
import numpy as np

from backtothecode_gym.envs.lib.board import ReadOnlyBoard
from backtothecode_gym.envs.lib.renderer import PrintRenderer
from backtothecode_gym.envs import BackToTheCodeEnv, BackToTheCodeEnvParams

class Player:
    def __init__(self):
      pass

    def reset(self, id, board : ReadOnlyBoard):
      self.id = id
      self.board = board

    def move(self):
      raise NotImplementedError()

class RandomPlayer(Player):
    def move(self):
        possible_actions = list(self.board.get_possible_actions(self.id))
        return np.random.choice(possible_actions)
    
# create the cartpole environment
env = gym.make('BackToTheCode', 
               board_size=(5, 5),
               opponent=RandomPlayer(), 
               renderer=PrintRenderer())

hero = RandomPlayer()

for episode in range(1):
  env.reset()
  hero.reset(BackToTheCodeEnvParams.HERO_ID, env.get_board())
  terminated = False
  truncated = False
  while not terminated and not truncated:
    env.render()
    action = hero.move()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()