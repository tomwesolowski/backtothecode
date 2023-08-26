import gymnasium as gym
# import importlib
import numpy as np

from time import sleep

import backtothecode_gym
import backtothecode_gym.envs.lib.board as board_lib
import backtothecode_gym.envs.lib.renderer as renderer_lib
import backtothecode_players as players_lib

from backtothecode_gym.envs import BackToTheCodeEnvParams

# importlib.reload(gym)
# importlib.reload(players_lib)
# importlib.reload(board_lib)
# importlib.reload(renderer_lib)

def create_game(params):
    players = [
        players_lib.RandomPlayer(
            id=params.HERO_ID,
            name='Hero',
            momentum=0.5
        ),
        players_lib.RandomPlayer(
            id=params.OPPONENT_ID,
            name='Opponent'
        )
    ]
    bttc = gym.make(
        id='BackToTheCode',
        players=players,
        renderer=renderer_lib.NoRenderer()
    )
    return bttc, players

bttc, players = create_game(BackToTheCodeEnvParams)
hero, opponent = players

for game_id in range(10):
    print(f"Game {game_id}...")
    bttc.reset()
    terminated = False
    truncated = False
    while not terminated and not truncated:
        old_board = bttc.unwrapped.board.copy()
        action = hero.move(old_board)
        observation, reward, terminated, truncated, info = bttc.step(action)
        new_board = bttc.unwrapped.board.copy()
        # Send feedback to the hero
        for player in players:
            player.feedback(old_board, action, reward, new_board, terminated or truncated)
        bttc.render()
        # sleep(0.1)

bttc.close()