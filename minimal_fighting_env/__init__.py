from gymnasium.envs.registration import register

register(
     id="MinimalFightingEnv-v0",
     entry_point="minimal_fighting_env.env:MinimalFightingEnv",
     max_episode_steps=1000,
)
