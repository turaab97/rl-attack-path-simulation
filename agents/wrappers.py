"""
wrappers.py
-----------
Shared Gymnasium wrappers used by both PPO and DQN agents.

Author: Syed Ali Turab
Course: MMAI 845 – Reinforcement Learning
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class IntActionWrapper(gym.Wrapper):
    """Coerce actions to Python int — required by nasim >= 0.10 with NumPy 2.x.

    NASim's step() expects a plain Python int, but SB3 may pass numpy
    integer types that NASim rejects.  This wrapper sits between the
    SB3 model and the environment to ensure type compatibility.
    """

    def step(self, action):
        return self.env.step(int(action))


class DenseRewardWrapper(gym.Wrapper):
    """Provide intermediate reward shaping to overcome NASim's sparse rewards.

    NASim only gives positive reward when a *sensitive* host is first
    compromised.  All other actions return -cost.  This means the agent
    must discover a 6+ step attack chain through 4 subnets before seeing
    any positive signal -- nearly impossible via random exploration.

    This wrapper detects progress by comparing consecutive observations.
    Any observation change (new host discovered, access gained, service
    revealed) receives a small bonus, creating a learnable gradient toward
    successful exploitation chains.

    Parameters
    ----------
    env : gym.Env
        A NASim environment (optionally wrapped with IntActionWrapper).
    progress_bonus : float
        Reward added when an action causes the observation to change.
    """

    def __init__(self, env: gym.Env, progress_bonus: float = 3.0) -> None:
        super().__init__(env)
        self.progress_bonus = progress_bonus
        self._prev_obs: np.ndarray | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs.copy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self._prev_obs is not None:
            diff = np.sum(np.abs(obs.astype(np.float32) - self._prev_obs.astype(np.float32)))
            if diff > 1e-6:
                reward += self.progress_bonus

        self._prev_obs = obs.copy()
        return obs, reward, terminated, truncated, info
