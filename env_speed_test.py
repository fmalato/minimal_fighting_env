import time

import minimal_fighting_env
import gymnasium as gym

import numpy as np
import pygame


if __name__ == '__main__':
    test_episodes = 10
    reward_shape = {
        "win": 1.0,
        "lose": -1.0,
        "hit": 0.2,
        "hurt": -0.2,
        "stun": -0.1,
        "block": 0.1,
        "time": -0.001
    }
    env = gym.make("MinimalFightingEnv-v0", render_mode=None, reward_shape=reward_shape, raw_pixel_obs=True, initial_health=7)

    step_times = []
    for i in range(test_episodes):
        print(f"Episode {i + 1}")
        obs, info = env.reset(seed=np.random.randint(0, 100000))
        terminated = False
        truncated = False
        timestep = 0
        rewards = [0.0, 0.0]
        while not (terminated or truncated):
            action_1 = np.random.randint(0, 6)
            action_2 = np.random.randint(0, 6)
            start = time.time()
            obs, rewards, terminated, truncated, info = env.step([action_1, action_2])
            step_times.append(time.time() - start)
            timestep += 1

        env.close()

    step_times = np.array(step_times)
    print(f"Average step time: {np.mean(step_times)} +/- {np.std(step_times)}")
    print(f"Average FPS: {int(1. / np.mean(step_times))} +/- {int(1. / np.std(step_times))}")