import gymnasium as gym
import numpy as np

from gymnasium.spaces import Discrete, MultiDiscrete

from .lib.board import Board, ReadOnlyBoard
from .lib.renderer import Renderer


class BackToTheCodeEnvParams():
    HEIGHT = 20
    WIDTH = 35
    HERO_ID = 0
    OPPONENT_ID = 1
    MAX_NUM_ROUNDS = 350
    NUM_ACTIONS = 4


class BackToTheCodeEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, players, renderer, **kwargs):
        super().__init__()
        self._board = Board(BackToTheCodeEnvParams.HEIGHT, BackToTheCodeEnvParams.WIDTH)
        self._players = players
        self.hero = self._players[BackToTheCodeEnvParams.HERO_ID]  
        self.opponent = self._players[BackToTheCodeEnvParams.OPPONENT_ID]   
        self.renderer = renderer 
        self.action_space = Discrete(4) # WENS
        self.observation_space = self.hero.get_observation_space()
        self.move_failure_reward = -100
        self.reset()

    def step(self, action):
        opponent_action = self.opponent.move(self._board)
        actions = [action, opponent_action]
        for player, action in zip(self._players, actions):
            self._board.add_to_buffer(player.id, action)
        # We assume that reward is the number of cells owned in this round.
        move_failures, rewards = self._board.finish_round()
        if any(move_failures):
            for id, reward in enumerate(rewards):
                if move_failures[id]:
                    rewards[id] = self.move_failure_reward
        for player, reward in zip(self._players, rewards):
            player.score += reward
        # All moves are OK
        self.round_number += 1
        min_score = min(player.score for player in self._players)
        max_score = max(player.score for player in self._players)
        done = (
            self.round_number >= BackToTheCodeEnvParams.MAX_NUM_ROUNDS or
            self._board.num_empty_cells() < max_score - min_score
        )
        return self.hero.observe(self._board), rewards[BackToTheCodeEnvParams.HERO_ID], done, False, {}
    
    def _draw_random_positions(self):
        return (
            np.random.choice(self._board.shape[0]),
            np.random.choice(self._board.shape[1])
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

    def reset(self, seed=None, options=None, is_training=False):
        self._board.reset(self._pick_initial_positions())
        for p in self._players:
            p.reset(is_training)
        self.round_number = 1
        return self.hero.observe(self._board), {}

    def render(self):
        self.renderer.render(ReadOnlyBoard(self._board), self._players, self.round_number)

    def seed(self, x):
        pass

    @property
    def board(self):
        return ReadOnlyBoard(self._board)

    def get_player(self, id):
        return self._players[id]