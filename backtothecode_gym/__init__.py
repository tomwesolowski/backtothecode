from gymnasium.envs.registration import register

register(
    id='BackToTheCode',
    entry_point='backtothecode_gym.envs:BackToTheCodeEnv',
)