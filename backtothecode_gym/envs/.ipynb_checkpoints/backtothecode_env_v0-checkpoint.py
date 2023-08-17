import gymnasium as gym
import numpy as np

from gymnasium.spaces import Discrete, MultiDiscrete


class BackToTheCodeEnv_v0(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()

        # dimensions of the grid
        self.width = 35
        self.height = 20

        # there are 5 possible actions: move N,E,S,W or stay in same state
        self.action_space = Discrete(4)

        # the observation will be the coordinates of Baby Robot
        self.observation_space = MultiDiscrete([self.width, self.height])

        # Baby Robot's position in the grid
        self.x = 0
        self.y = 0

    def step(self, action):
        obs = np.array([self.x,self.y])
        reward = -1
        terminated = True
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset Baby Robot's position in the grid
        self.x = 0
        self.y = 0
        info = {}
        return np.array([self.x,self.y]),info

    def render(self):
        pass