import minimal_fighting_env
import gymnasium as gym

import numpy as np
import pygame


actions = {
    'a': 1,
    'd': 2,
    'i': 3,
    'k': 4,
    'o': 5,
    'l': 6
}


def handle_inputs():
    """Control the agent using a/d keys"""
    action = 0
    close_request = False
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            try:
                if event.unicode in actions.keys():
                    action = int(actions[event.unicode])
                    break
                elif event.unicode == '\x1b':
                    close_request = True
            except KeyError:
                print("invalid input")

    return action, close_request


if __name__ == '__main__':
    env = gym.make("MinimalFightingEnv-v0", render_mode="human")
    obs, info = env.reset(seed=np.random.randint(0, 100000))
    terminated = False
    truncated = False
    timestep = 0
    rewards = [0.0, 0.0]
    while not (terminated or truncated):
        action_1, close_request = handle_inputs()
        #action_2 = 5 if timestep % 2 == 0 else 3
        action_2 = np.random.randint(0, 6)
        if close_request:
            break
        obs, rewards, terminated, truncated, info = env.step([action_1, action_2])
        timestep += 1
    print(rewards)

    env.close()
