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


class ActionMaskWrapper(gym.Wrapper):
    """Expose NASim's action validity mask for use with sb3-contrib MaskablePPO.

    NASim 0.12 has a ``get_action_mask()`` method but it contains a bug
    (calls ``host_discovered`` on the Network object instead of the State
    object).  This wrapper fixes the mask computation and exposes it via
    the ``action_masks()`` method that MaskablePPO expects.

    An action is valid when the target host has been discovered.  This
    reduces the effective action space from 110 to ~5-15 valid actions
    per step, making exploration tractable.
    """

    def action_masks(self) -> np.ndarray:
        nasim_env = self.unwrapped
        mask = np.zeros(nasim_env.action_space.n, dtype=np.float32)
        state = nasim_env.current_state
        for a_idx in range(nasim_env.action_space.n):
            action = nasim_env.action_space.get_action(a_idx)
            target = action.target
            if state.host_discovered(target) and state.host_reachable(target):
                mask[a_idx] = 1.0
        if mask.sum() == 0:
            mask[:] = 1.0
        return mask

    def step(self, action):
        return self.env.step(int(action))


class DenseRewardWrapper(gym.Wrapper):
    """Provide intermediate reward shaping to overcome NASim's sparse rewards.

    NASim only gives positive reward when a *sensitive* host is first
    compromised.  All other actions return -cost.  This wrapper detects
    progress by comparing consecutive observations.  Actions that cause
    multiple observation features to change (host discovered, access
    gained) receive a small bonus.

    NASim updates a step/action indicator feature on every step (exactly
    1 feature changes).  Real progress changes 2+ features.

    Parameters
    ----------
    env : gym.Env
        A NASim environment (optionally wrapped with IntActionWrapper).
    progress_bonus : float
        Reward added when an action causes 2+ observation features to change.
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
            n_changed = int(
                np.sum(np.abs(obs.astype(np.float32) - self._prev_obs.astype(np.float32)) > 1e-6)
            )
            if n_changed > 1:
                reward += self.progress_bonus

        self._prev_obs = obs.copy()
        return obs, reward, terminated, truncated, info
