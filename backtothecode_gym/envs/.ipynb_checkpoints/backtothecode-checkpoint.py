import gymnasium as gym
from gymnasium.spaces import Discrete

class BackToTheCodeEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, grid_size=[15,15], 
                 unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, 
                 n_foods=1, random_init=True, **kwargs):
        super().__init__()
        self.grid_size = grid_size
        self.unit_size = unit_size
        self.unit_gap = unit_gap
        self.snake_size = snake_size
        self.n_snakes = n_snakes
        self.n_foods = n_foods
        self.viewer = None
        self.action_space = Discrete(4)
        self.random_init = random_init

    def step(self, action):
        self.last_obs, rewards, done, info = self.controller.step(action)
        return self.last_obs, rewards, done, info

    def reset(self):
        self.controller = Controller(self.grid_size, self.unit_size, self.unit_gap, self.snake_size, self.n_snakes, self.n_foods, random_init=self.random_init)
        self.last_obs = self.controller.grid.grid.copy()
        return self.last_obs

    def render(self, mode='human', close=False, frame_speed=.1):
        if self.viewer is None:
            self.fig = plt.figure()
            self.viewer = self.fig.add_subplot(111)
            plt.ion()
            self.fig.show()
        else:
            self.viewer.clear()
            self.viewer.imshow(self.last_obs)
            plt.pause(frame_speed)
        self.fig.canvas.draw()

    def seed(self, x):
        pass