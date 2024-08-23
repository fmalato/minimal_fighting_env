from typing import Optional

import gymnasium as gym
import numpy as np
import pygame

from player.player import Player


REQUIRED_REWARD_CONDITIONS = ["win", "lose", "hit", "hurt", "block", "stun", "time"]
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
P1_COLOR = (204, 0, 0)
P2_COLOR = (0, 102, 204)
P1_BLOCK_COLOR = (255, 153, 153)
P2_BLOCK_COLOR = (153, 255, 255)
GREY = (194, 194, 194)
WHITE = (255, 255, 255)


class MinimalFightingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 3}

    def __init__(self, max_timesteps: int = 1000, initial_health: int = 3, reward_shape: Optional[dict] = None, raw_pixel_obs: bool = False, render_mode: str = None):
        super().__init__()
        # TODO: [TBD] add best of series?
        if reward_shape is not None:
            assert set(list(reward_shape.keys())) == set(REQUIRED_REWARD_CONDITIONS), "Missing conditions %s for reward specification".format(set(REQUIRED_REWARD_CONDITIONS) - set(list(reward_shape.keys())))
        self.p1 = Player(color=P1_COLOR, max_hp=initial_health)
        self.p2 = Player(color=P2_COLOR, max_hp=initial_health)
        self.last_frame = None
        self.grid_width = 20
        self.grid_height = 12
        self.pixel_render_size = 25
        self.hitbox_size = 2
        self.stun_steps = 5
        self.damaged_steps = 4
        self.p1_start_pos = (2, self.grid_height - 2)
        self.p2_start_pos = (self.grid_width - 3, self.grid_height - 2)
        self.p1_last_action = 0
        self.p2_last_action = 0

        self.timestep = 0
        self.max_timesteps = max_timesteps
        self.initial_health = initial_health
        self.reward_shape = reward_shape if reward_shape else self._create_default_reward()
        self.raw_pixel_obs = raw_pixel_obs

        if self.raw_pixel_obs:
            obs_dict = {
                "p1": gym.spaces.Box(low=0.0, high=1.0, shape=(self.grid_height, self.grid_width, 3)),
                "p2": gym.spaces.Box(low=0.0, high=1.0, shape=(self.grid_height, self.grid_width, 3))
            }
        else:
            obs_dict = {
                "p1": gym.spaces.Box(low=0.0, high=30.0, shape=(2 * DATA_OBS_DIM,), dtype=np.int32),
                "p2": gym.spaces.Box(low=0.0, high=30.0, shape=(2 * DATA_OBS_DIM,), dtype=np.int32)
            }
        self.observation_space = gym.spaces.Dict(spaces=obs_dict)

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
            "hit": 0.0,
            "hurt": 0.0,
            "block": 0.0,
            "stun": 0.0,
            "time": 0.0
        }

        return reward

    def reset(self, seed: int = 42):
        super().reset(seed=seed)
        self.p1.reset(x=self.p1_start_pos[0], y=self.p1_start_pos[1])
        self.p1_last_action = 0
        self.p2.reset(x=self.p2_start_pos[0], y=self.p2_start_pos[1])
        self.p2_last_action = 0

        obs = self._get_obs()
        info = self._get_info(p1_reward=None, p2_reward=None)

        self.timestep = 0

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(self, actions: list[int]):
        terminated = False
        truncated = False
        self.p1_last_action = actions[0]
        self.p2_last_action = actions[1]

        self._move_players()

        if self.p1.get_damaged() > 0:
            self.p1.decrease_damaged()
        if self.p2.get_damaged() > 0:
            self.p2.decrease_damaged()

        p1_hit, p2_hit = self._check_collisions(self.p1.get_position(), self.p2.get_position())
        if p1_hit:
            p2_pos = self.p2.get_position()
            self.p2.set_hp(self.p2.get_hp() - 1)
            self.p2.set_damaged(self.damaged_steps)
            self.p2.set_position(x=min(p2_pos["x"] + 2, self.grid_width - 1), y=p2_pos["y"])
        elif p2_hit:
            p1_pos = self.p1.get_position()
            self.p1.set_hp(self.p1.get_hp() - 1)
            self.p1.set_damaged(self.damaged_steps)
            self.p1.set_position(x=max(0, p1_pos["x"] - 2), y=p1_pos["y"])

        p1_dead = self.p1.get_hp() == 0
        p2_dead = self.p2.get_hp() == 0

        obs = self._get_obs()

        if p1_dead or p2_dead:
            terminated = True
        if self.timestep >= self.max_timesteps:
            truncated = True

        p1_reward, p2_reward = self.compute_rewards(p1_hit, p2_hit, p1_dead, p2_dead)
        rewards = [float(np.sum(list(p1_reward.values()))) - self.reward_shape["time"], float(np.sum(list(p2_reward.values()))) - self.reward_shape["time"]]

        info = self._get_info(p1_reward, p2_reward)

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
        # Stun/Block condition
        if self.p1.get_stunned() == self.stun_steps:
            p1_rewards["stun"] += self.reward_shape["stun"]
            p2_rewards["block"] += self.reward_shape["block"]
        if self.p2.get_stunned() == self.stun_steps:
            p1_rewards["block"] += self.reward_shape["block"]
            p2_rewards["stun"] += self.reward_shape["stun"]

        return p1_rewards, p2_rewards

    def _move_players(self):
        p1_pos = self.p1.get_position()
        p2_pos = self.p2.get_position()

        p1_stun = self.p1.get_stunned()
        p2_stun = self.p2.get_stunned()

        if p1_stun == 0:
            if ACTION_NAMES[self.p1_last_action] == "left":
                self.p1.set_position(x=max(0, min(p1_pos["x"] - 1, p2_pos["x"] - 1)), y=p1_pos["y"])
            elif ACTION_NAMES[self.p1_last_action] == "right":
                self.p1.set_position(x=max(0, min(p1_pos["x"] + 1, p2_pos["x"] - 1)), y=p1_pos["y"])
        else:
            self.p1.decrease_stunned()

        if p2_stun == 0:
            if ACTION_NAMES[self.p2_last_action] == "left":
                self.p2.set_position(x=min(self.grid_width - 1, max(p1_pos["x"] + 1, p2_pos["x"] - 1)), y=p2_pos["y"])
            elif ACTION_NAMES[self.p2_last_action] == "right":
                self.p2.set_position(x=min(self.grid_width - 1, max(p1_pos["x"] + 1, p2_pos["x"] + 1)), y=p2_pos["y"])
        else:
            self.p2.decrease_stunned()

    def _check_collisions(self, p1_pos, p2_pos):
        p1_hit = False
        p2_hit = False
        if p2_pos["x"] - p1_pos["x"] <= self.hitbox_size:
            # Can hit at the same time
            if ACTION_NAMES[self.p1_last_action] == "punch":
                if ACTION_NAMES[self.p2_last_action] != "block_high":
                    p1_hit = True
                    self.p2.set_stunned(0)
                else:
                    self.p1.set_stunned(self.stun_steps)
            elif ACTION_NAMES[self.p1_last_action] == "kick":
                if ACTION_NAMES[self.p2_last_action] != "block_low":
                    p1_hit = True
                    self.p2.set_stunned(0)
                else:
                    self.p1.set_stunned(self.stun_steps)
            if ACTION_NAMES[self.p2_last_action] == "punch":
                if ACTION_NAMES[self.p1_last_action] != "block_high":
                    p2_hit = True
                    self.p1.set_stunned(0)
                else:
                    self.p2.set_stunned(self.stun_steps)
            elif ACTION_NAMES[self.p2_last_action] == "kick":
                if ACTION_NAMES[self.p1_last_action] != "block_low":
                    p2_hit = True
                    self.p1.set_stunned(0)
                else:
                    self.p2.set_stunned(self.stun_steps)
            # Invulnerable for some time after being hit
            if self.p1.get_damaged() > 0:
                p2_hit = False
            if self.p2.get_damaged() > 0:
                p1_hit = False

        return p1_hit, p2_hit

    def _get_obs(self):
        p1_state = self.p1.get_state()
        p1_state.append(self.p1_last_action)
        p2_state = self.p2.get_state()
        p2_state.append(self.p2_last_action)
        if self.raw_pixel_obs:
            obs = self._build_obs_grid(p1_state, p2_state)
        else:
            obs = {
                "p1": np.concatenate([p1_state, p2_state]).astype(np.int32),
                "p2": np.concatenate([p2_state, p1_state]).astype(np.int32)
            }

        return obs

    def _get_info(self, p1_reward, p2_reward):
        p1_state = self.p1.get_dict_state()
        p2_state = self.p2.get_dict_state()
        info = {
            "p1_state": p1_state,
            "p2_state": p2_state,
            "p1_reward": p1_reward if p1_reward else "None",
            "p2_reward": p2_reward if p2_reward else "None"
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
        p1_pos = self.p1.get_position()
        p2_pos = self.p2.get_position()
        p1_color = self.p1.get_color()
        p2_color = self.p2.get_color()
        p1_stun = self.p1.get_stunned()
        p2_stun = self.p2.get_stunned()

        for player, pos, c in zip([self.p1, self.p2], [p1_pos, p2_pos], [p1_color, p2_color]):
            player_damaged = player.get_damaged()
            pygame.draw.rect(
                canvas,
                GREY if (player_damaged > 0 and player_damaged % 2 == 0) else c,
                pygame.Rect(
                    pos["x"] * self.pixel_render_size, pos["y"] * self.pixel_render_size, self.pixel_render_size, 2 * self.pixel_render_size
                )
            )

        # Plot lives
        for i in range(self.p1.get_hp()):
            pygame.draw.rect(
                canvas,
                p1_color,
                pygame.Rect(
                    i * self.pixel_render_size, 0, self.pixel_render_size, self.pixel_render_size
                )
            )
        for i in range(self.grid_width - 1, self.grid_width - self.p2.get_hp() - 1, -1):
            pygame.draw.rect(
                canvas,
                p2_color,
                pygame.Rect(
                    i * self.pixel_render_size, 0, self.pixel_render_size, self.pixel_render_size
                )
            )

        # Plot stunned
        for i in range(p1_stun):
            pygame.draw.rect(
                canvas,
                WHITE,
                pygame.Rect(
                    i * self.pixel_render_size, self.pixel_render_size, self.pixel_render_size, self.pixel_render_size
                )
            )
        for i in range(self.grid_width - 1, self.grid_width - p2_stun - 1, -1):
            pygame.draw.rect(
                canvas,
                WHITE,
                pygame.Rect(
                    i * self.pixel_render_size, self.pixel_render_size, self.pixel_render_size, self.pixel_render_size
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

        # Draw attacks
        for p, c, a, o, d in zip([p1_pos, p2_pos], [p1_color, p2_color], [self.p1_last_action, self.p2_last_action], [1, -1], [0, int(self.pixel_render_size / 2)]):
            if ACTION_NAMES[a] == "punch":
                pygame.draw.rect(
                    canvas,
                    c,
                    pygame.Rect(
                        (p["x"] + o) * self.pixel_render_size + d, p["y"] * self.pixel_render_size, int(self.pixel_render_size / 2), self.pixel_render_size
                    )
                )
            elif ACTION_NAMES[a] == "kick":
                pygame.draw.rect(
                    canvas,
                    c,
                    pygame.Rect(
                        (p["x"] + o) * self.pixel_render_size + d, (p["y"] + 1) * self.pixel_render_size, int(self.pixel_render_size / 2), self.pixel_render_size
                    )
                )

        # Draw parry
        for p, c, a, o, d in zip([p1_pos, p2_pos], [P1_BLOCK_COLOR, P2_BLOCK_COLOR], [self.p1_last_action, self.p2_last_action], [1, -1], [0, int(self.pixel_render_size / 2)]):
            if ACTION_NAMES[a] == "block_high":
                pygame.draw.rect(
                    canvas,
                    c,
                    pygame.Rect(
                        (p["x"] + o) * self.pixel_render_size + d, p["y"] * self.pixel_render_size, int(self.pixel_render_size / 2), self.pixel_render_size
                    )
                )
            elif ACTION_NAMES[a] == "block_low":
                pygame.draw.rect(
                    canvas,
                    c,
                    pygame.Rect(
                        (p["x"] + o) * self.pixel_render_size + d, (p["y"] + 1) * self.pixel_render_size, int(self.pixel_render_size / 2), self.pixel_render_size
                    )
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
        p1_obs = np.zeros(shape=(self.grid_height, self.grid_width, 3), dtype=np.float32)
        p1_color = self.p1.get_color()
        p2_color = self.p2.get_color()

        # Player
        p1_obs[p1_state[1]: p1_state[1] + 2, p1_state[0]] = np.array(p1_color)
        p1_obs[p2_state[1]: p2_state[1] + 2, p2_state[0]] = np.array(p2_color)
        # HP
        p1_obs[0, 0: p1_state[2]] = np.array(p1_color)
        p1_obs[0, self.grid_width - p2_state[2]: self.grid_width] = np.array(p2_color)
        # Stun
        p1_obs[0, 0: p1_state[3]] = np.array(WHITE)
        p1_obs[0, self.grid_width - p2_state[3]: self.grid_width] = np.array(WHITE)
        # Action
        for s, c, bc, o in zip([p1_state, p2_state], [p1_color, p2_color], [P1_BLOCK_COLOR, P2_BLOCK_COLOR], [1, -1]):
            if ACTION_NAMES[p1_state[4]] == "punch":
                p1_obs[s[1], s[0] + o] = np.array(c)
            elif ACTION_NAMES[p1_state[4]] == "kick":
                p1_obs[s[1] + 1, s[0] + o] = np.array(c)
            if ACTION_NAMES[p1_state[4]] == "block_high":
                p1_obs[s[1], s[0] + o] = np.array(bc)
            elif ACTION_NAMES[p1_state[4]] == "block_low":
                p1_obs[s[1] + 1, s[0] + o] = np.array(bc)
        # Damaged


        # p2_obs from p1_obs by inverting colors
        p2_obs = np.zeros_like(p1_obs)
        # TODO: find more robust way: if player colors change, this will likely break
        # Swap main color
        p2_obs[np.where(p1_obs[:, :, 0] == p1_color[0])] = np.array(p2_color)
        p2_obs[np.where(p1_obs[:, :, 2] == p2_color[2])] = np.array(p1_color)
        # Swap block color
        p2_obs[np.where(p1_obs[:, :, 2] == P1_BLOCK_COLOR[2])] = np.array(P2_BLOCK_COLOR)
        p2_obs[np.where(p1_obs[:, :, 0] == P2_BLOCK_COLOR[0])] = np.array(P1_BLOCK_COLOR)
        p2_obs = np.flip(p2_obs, axis=1)

        obs_dict = {
            "p1": p1_obs / 255.,
            "p2": p2_obs / 255.
        }

        return obs_dict

    def one_hot_act(self, action):
        one_hot_act = np.zeros_like(ACTION_NAMES)
        one_hot_act[action] = 1
