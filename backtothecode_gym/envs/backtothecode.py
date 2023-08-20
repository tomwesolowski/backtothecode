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

    def __init__(self, players, renderer, board_size=[20, 35], **kwargs):
        super().__init__()

        self.board_size = board_size
        self.board_height, self.board_width = self.board_size
        self._players = players
        self.opponent = self._players[BackToTheCodeEnvParams.OPPONENT_ID]       
        self.renderer = renderer
        self.action_space = Discrete(4) # WENS
        self.observation_space = MultiDiscrete(
            self.board_size + # hero's position
            self.board_size # opponent's position
        )

    def step(self, action):
        opponent_action = self.opponent.move()
        self._board.add_to_buffer(BackToTheCodeEnvParams.HERO_ID, action)
        self._board.add_to_buffer(BackToTheCodeEnvParams.OPPONENT_ID, opponent_action)
        observation, rewards, truncated = self._board.finish_round()
        done = (
            self.round_number >= BackToTheCodeEnvParams.MAX_NUM_ROUNDS or
            self._board.num_empty_cells() == 0
        )
        if truncated:
            return observation, 0, done, truncated, {}
        self.round_number += 1
        for player, reward in zip(self._players, rewards):
            player.score += reward
        return observation, rewards[BackToTheCodeEnvParams.HERO_ID], done, truncated, {}
    
    def _draw_random_positions(self):
        return (
            np.random.choice(self.board_height),
            np.random.choice(self.board_width)
        )

    def _pick_initial_positions(self):
        initial_positions = []
        for _ in self._players:
            while True:
                position = self._draw_random_positions()
                if position not in initial_positions:
                    initial_positions.append(position)
                    break
        return initial_positions

    def reset(self, seed=None, options=None):
        self._board = Board(*self.board_size, self._pick_initial_positions())
        self.renderer.reset(self.board)
        self.round_number = 1
        return self._board.get_observations(), {}

    def render(self):
        self.renderer.render(self.round_number, self._players)

    def seed(self, x):
        pass

    @property
    def board(self):
        return ReadOnlyBoard(self._board)

    def get_player(self, id):
        return self._players[id]