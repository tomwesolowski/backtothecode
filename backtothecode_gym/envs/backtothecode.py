import gymnasium as gym
import numpy as np

from gymnasium.spaces import Discrete, MultiDiscrete

from .lib.board import Board, ReadOnlyBoard
from .lib.renderer import Renderer


class BackToTheCodeEnvParams():
    HERO_ID = 0
    OPPONENT_ID = 1
    MAX_NUM_ROUNDS = 350


class BackToTheCodeEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, opponent, renderer, board_size=[35, 20], **kwargs):
        super().__init__()

        self.num_players = 2
        self.board_size = board_size
        self.board_height, self.board_width = self.board_size
        self.opponent = opponent       
        self.renderer = renderer
        self.action_space = Discrete(4) # WENS
        self.observation_space = MultiDiscrete(
            self.board_size + # hero's position
            self.board_size # opponent's position
        )

    def step(self, action):
        opponent_action = self.opponent.move()
        self.board.add_to_buffer(BackToTheCodeEnvParams.HERO_ID, action)
        self.board.add_to_buffer(BackToTheCodeEnvParams.OPPONENT_ID, opponent_action)
        observation, (hero_reward, _), truncated = self.board.finish_round()
        done = (
            self.round_number >= BackToTheCodeEnvParams.MAX_NUM_ROUNDS or
            self.board.num_empty_cells() == 0
        )
        self.round_number += 1
        return observation, hero_reward, done, truncated, {}
    
    def _draw_random_positions(self):
        return (
            np.random.choice(self.board_height),
            np.random.choice(self.board_width)
        )

    def _pick_initial_positions(self):
        initial_positions = []
        for _ in range(self.num_players):
            while True:
                position = self._draw_random_positions()
                if position not in initial_positions:
                    initial_positions.append(position)
                    break
        return initial_positions

    def reset(self, seed=None, options=None):
        self.board = Board(*self.board_size, self._pick_initial_positions())
        self.opponent.reset(BackToTheCodeEnvParams.OPPONENT_ID, self.get_board())
        self.renderer.reset(self.get_board())
        self.round_number = 1
        return self.board.get_observations(), {}

    def render(self):
        self.renderer.render(self.round_number)

    def seed(self, x):
        pass

    def get_board(self):
        return ReadOnlyBoard(self.board)
        