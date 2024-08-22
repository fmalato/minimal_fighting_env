import minimal_fighting_env
import gymnasium as gym

import numpy as np
import pygame


actions = {
    'a': 1,
    'd': 2,
    'e': 3,
    'q': 4,
    'r': 5,
    'f': 6
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
    while not (terminated or truncated):
        action_1, close_request = handle_inputs()
        action_2 = np.random.randint(0, 6)
        if close_request:
            break
        obs, reward, terminated, truncated, info = env.step([action_1, action_2])

    env.close()
