import gymnasium as gym
import numpy as np

from backtothecode_gym.envs.lib.renderer import PrintRenderer
from backtothecode_gym.envs import BackToTheCodeEnv, BackToTheCodeEnvParams
from backtothecode_players import RandomPlayer, KeyboardPlayer

envs.registration.register(
    id='BackToTheCode',
    entry_point='backtothecode_gym.envs:BackToTheCodeEnv',
)

env = gym.make('BackToTheCode', 
               board_size=(5, 5),
               opponent=RandomPlayer(), 
               renderer=PrintRenderer())

hero = KeyboardPlayer()

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