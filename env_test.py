import time

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

ENV_RULES = """
----------------------------------------------------------------------
------------------------ MINIMAL FIGHTING ENV ------------------------
----------------------------------------------------------------------

You are playing as the RED character.

1. CONTROLS
                           [ KEYBOARD ]
--------------------------------------[punch]---[block punch]------
[move left]----------------------------------- I - O --------------
---- A --- D ---------------------------------- K - L -------------
---------[move right]-------------------[kick]-------[block kick]--
-------------------------------------------------------------------

- Move your character left/right with A/D keys.
- Attack with a punch by pressing I.
- Attack with a kick by pressing K.
- Block an incoming punch with O.
- Block an incoming kick with L.

2. RULES
- Each player has 3 Hit Points.
- If the opponent hits you, you lose 1 Hit Point.
- Successfully blocking an incoming attack stuns the opponent for 
  5 time steps.
- If a stunned player is hit, they are no longer stunned.
- When your Hit Points reach 0, you die.
- Last player standing wins.
"""

FIGHT = """
============================
  __ _       _     _     _
 / _(_)     | |   | |   | |
| |_ _  __ _| |__ | |_  | |
|  _| |/ _` | '_ \| __| | |
| | | | (_| | | | | |_  |_|
|_| |_|\__, |_| |_|\__|  
        __/ |           (_)
       |___/  
============================
"""

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


def rules_and_countdown(countdown=10):
    print(ENV_RULES)
    time.sleep(countdown - 3)
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print(FIGHT)


if __name__ == '__main__':
    reward_shape = {
        "win": 1.0,
        "lose": -1.0,
        "hit": 0.2,
        "hurt": -0.2,
        "stun": -0.1,
        "block": 0.1
    }
    env = gym.make("MinimalFightingEnv-v0", render_mode="human", reward_shape=reward_shape)
    obs, info = env.reset(seed=np.random.randint(0, 100000))
    terminated = False
    truncated = False
    timestep = 0
    rewards = [0.0, 0.0]
    rules_and_countdown()
    while not (terminated or truncated):
        action_1, close_request = handle_inputs()
        action_2 = np.random.randint(0, 6)
        if close_request:
            break
        obs, rewards, terminated, truncated, info = env.step([action_1, action_2])
        timestep += 1
        print(rewards)

    env.close()
