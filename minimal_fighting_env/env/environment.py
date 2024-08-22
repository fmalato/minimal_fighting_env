from typing import Optional

import gymnasium as gym
import numpy as np
import pygame

from player.player import Player


REQUIRED_REWARD_CONDITIONS = ["win", "lose", "win_best_of", "hit", "hurt"]
DATA_OBS_DIM = 6

ACTION_TEMPLATE = {
    "noop": 0,
    "left": 0,
    "right": 0,
    "punch": 0,
    "kick": 0,
    "block_high": 0,
    "block_low": 0
}

ACTION_NAMES = ["noop", "left", "right", "punch", "kick", "block_high", "block_low"]


class MinimalFightingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, max_timesteps: int = 1000, initial_health: int = 3, best_of: int = 1, reward_shape: Optional[dict] = None, raw_pixel_obs: bool = False, render_mode: str = None):
        super().__init__()

        if reward_shape is not None:
            assert set(list(reward_shape.keys())) == set(REQUIRED_REWARD_CONDITIONS), "Missing conditions %s for reward specification".format(set(REQUIRED_REWARD_CONDITIONS) - set(list(reward_shape.keys())))
        self.p1 = Player(color=(204, 0, 0), max_hp=initial_health)
        self.p2 = Player(color=(0, 102, 204), max_hp=initial_health)
        self.last_frame = None
        self.grid_width = 25
        self.grid_height = 15
        self.pixel_render_size = 25
        self.hitbox_size = 2
        self.p1_start_pos = (2, self.grid_height - 2)
        self.p2_start_pos = (self.grid_width - 3, self.grid_height - 2)

        self.timestep = 0
        self.max_timesteps = max_timesteps
        self.initial_health = initial_health
        self.best_of = best_of
        self.reward_shape = reward_shape if reward_shape else self._create_default_reward()
        self.raw_pixel_obs = raw_pixel_obs

        if self.raw_pixel_obs:
            obs_dict = {
                "p1": gym.spaces.Box(low=0.0, high=1.0, shape=(self.grid_height, self.grid_width, 3)),
                "p2": gym.spaces.Box(low=0.0, high=1.0, shape=(self.grid_height, self.grid_width, 3)),
                "hud": gym.spaces.Box(low=0.0, high=1.0, shape=(self.grid_height, self.grid_width, 3))
            }
            self.observation_space = gym.spaces.Dict(spaces=obs_dict)
        else:
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2 * DATA_OBS_DIM,))
        self.action_space = gym.spaces.Dict(spaces={
            "p1": gym.spaces.Discrete(n=len(ACTION_NAMES)),
            "p2": gym.spaces.Discrete(n=len(ACTION_NAMES))
        })

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.window_width = self.grid_width * self.pixel_render_size
        self.window_height = self.grid_height * self.pixel_render_size
        self.grid = np.zeros(shape=(self.grid_height, self.grid_width)) if self.raw_pixel_obs else None

    def _create_default_reward(self):
        reward = {
            "win": 1.0,
            "lose": -1.0,
            "win_best_of": self.best_of if self.best_of > 1 else 0.0,
            "hit": 0.0,
            "hurt": 0.0
        }

        return reward

    def reset(self, seed: int = 42):
        super().reset(seed=seed)
        # TODO: make it adaptable for v2?
        self.p1.reset(x=self.p1_start_pos[0], y=self.p1_start_pos[1])
        self.p2.reset(x=self.p2_start_pos[0], y=self.p2_start_pos[1])

        obs = self._get_obs()
        info = self._get_info()

        self.timestep = 0

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(self, actions: list[int]):
        terminated = False
        truncated = False
        p1_action = actions[0]
        p2_action = actions[1]

        self._move_players(p1_action, p2_action)

        p1_hit, p2_hit = self._check_collisions(self.p1.get_position(), self.p2.get_position(), p1_action, p2_action)
        if p1_hit:
            self.p2.set_hp(self.p2.get_hp() - 1)
        elif p2_hit:
            self.p1.set_hp(self.p1.get_hp() - 1)
        p1_dead = self.p1.get_hp() == 0
        p2_dead = self.p2.get_hp() == 0

        obs = self._get_obs()
        info = self._get_info()

        if p1_dead or p2_dead:
            terminated = True
        if self.timestep >= self.max_timesteps:
            truncated = True

        rewards = self.compute_rewards(p1_hit, p2_hit, p1_dead, p2_dead)

        self.timestep += 1

        if self.render_mode == "human":
            self._render_frame()

        return obs, rewards, terminated, truncated, info

    def compute_rewards(self, p1_hit, p2_hit, p1_dead, p2_dead):
        p1_rewards = dict(zip(REQUIRED_REWARD_CONDITIONS, [0.0 for _ in range(len(REQUIRED_REWARD_CONDITIONS))]))
        p2_rewards = dict(zip(REQUIRED_REWARD_CONDITIONS, [0.0 for _ in range(len(REQUIRED_REWARD_CONDITIONS))]))
        # Dead condition
        if p1_dead:
            p1_rewards["lose"] += self.reward_shape["lose"]
            p2_rewards["win"] += self.reward_shape["win"]
        elif p2_dead:
            p1_rewards["win"] += self.reward_shape["win"]
            p2_rewards["lose"] += self.reward_shape["lose"]
        # Hit condition
        if p1_hit:
            p1_rewards["hit"] += self.reward_shape["hit"]
            p2_rewards["hurt"] += self.reward_shape["hurt"]
        elif p2_hit:
            p1_rewards["hurt"] += self.reward_shape["hurt"]
            p2_rewards["hit"] += self.reward_shape["hit"]

        return [p1_rewards, p2_rewards]

    def _move_players(self, p1_action, p2_action):
        p1_pos = self.p1.get_position()
        p2_pos = self.p2.get_position()
        # TODO: double-check this
        if ACTION_NAMES[p1_action] == "left":
            self.p1.set_position(x=max(0, min(p1_pos["x"] - 1, p2_pos["x"])), y=p1_pos["y"])
        elif ACTION_NAMES[p1_action] == "right":
            self.p1.set_position(x=max(0, min(p1_pos["x"] + 1, p2_pos["x"])), y=p1_pos["y"])

        if ACTION_NAMES[p2_action] == "left":
            self.p2.set_position(x=min(self.grid_width - 1, max(p1_pos["x"] + 1, p2_pos["x"] - 1)), y=p2_pos["y"])
        elif ACTION_NAMES[p2_action] == "right":
            self.p2.set_position(x=min(self.grid_width - 1, max(p1_pos["x"] - 1, p2_pos["x"] - 1)), y=p2_pos["y"])

    def _check_collisions(self, p1_pos, p2_pos, p1_action, p2_action):
        p1_hit = False
        p2_hit = False
        if p2_pos["x"] - p1_pos["x"] <= self.hitbox_size:
            # Can hit at the same time
            if ACTION_NAMES[p1_action] == "punch":
                if ACTION_NAMES[p2_action] != "block_high":
                    p1_hit = True
            if ACTION_NAMES[p2_action] == "punch":
                if ACTION_NAMES[p1_action] != "block_high":
                    p2_hit = True
            if ACTION_NAMES[p1_action] == "kick":
                if ACTION_NAMES[p2_action] != "block_low":
                    p1_hit = True
            if ACTION_NAMES[p2_action] == "kick":
                if ACTION_NAMES[p1_action] != "block_low":
                    p2_hit = True

        return p1_hit, p2_hit

    def _get_obs(self):
        p1_state = self.p1.get_state()
        p2_state = self.p2.get_state()
        if self.raw_pixel_obs:
            obs = self._build_obs_grid(p1_state, p2_state)
        else:
            obs = [np.concatenate([p1_state, p2_state]), np.concatenate([p2_state, p1_state])]

        return obs

    def _get_info(self):
        p1_state = self.p1.get_dict_state()
        p2_state = self.p2.get_dict_state()
        info = {
            "p1": p1_state,
            "p2": p2_state
        }

        return info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((0, 0, 0))

        pygame.draw.rect(
            canvas,
            self.p1.get_color(),
            pygame.Rect(
                self.p1.get_position()["x"] * self.pixel_render_size, self.p1.get_position()["y"] * self.pixel_render_size, self.pixel_render_size, 2 * self.pixel_render_size
            )
        )
        pygame.draw.rect(
            canvas,
            self.p2.get_color(),
            pygame.Rect(
                self.p2.get_position()["x"] * self.pixel_render_size, self.p2.get_position()["y"] * self.pixel_render_size, self.pixel_render_size, 2 * self.pixel_render_size
            )
        )

        # Plot lives
        for i in range(self.p1.get_hp()):
            pygame.draw.rect(
                canvas,
                self.p1.get_color(),
                pygame.Rect(
                    i * self.pixel_render_size, 0, self.pixel_render_size, self.pixel_render_size
                )
            )
        for i in range(self.grid_width - 1, self.grid_width - self.p2.get_hp() - 1, -1):
            pygame.draw.rect(
                canvas,
                self.p2.get_color(),
                pygame.Rect(
                    i * self.pixel_render_size, 0, self.pixel_render_size, self.pixel_render_size
                )
            )

        # Plot grid
        for x in range(self.grid_height + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, self.pixel_render_size * x),
                (self.window_width, self.pixel_render_size * x),
                width=3,
            )
        for x in range(self.grid_width + 1):
            pygame.draw.line(
                canvas,
                0,
                (self.pixel_render_size * x, 0),
                (self.pixel_render_size * x, self.window_height),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _build_obs_grid(self, p1_state, p2_state):
        obs = []


    def one_hot_act(self, action):
        one_hot_act = np.zeros_like(ACTION_NAMES)
        one_hot_act[action] = 1
