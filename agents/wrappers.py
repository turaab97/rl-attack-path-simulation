"""
wrappers.py
-----------
Shared Gymnasium wrappers used by both PPO and DQN agents.

Author: Syed Ali Turab
Course: MMAI 845 -- Reinforcement Learning

This module contains three wrappers that are composed around the raw NASim
environment to make it compatible with Stable-Baselines3 and to improve
learning:

  1. IntActionWrapper   -- type coercion (NASim requires Python int)
  2. ActionMaskWrapper  -- exposes valid-action masks for MaskablePPO / DQN
  3. DenseRewardWrapper -- adds intermediate reward shaping for sparse-reward
                           environments

The standard wrapper composition order used during training is:

  raw NASim env
    -> IntActionWrapper        (fix action dtype)
      -> ActionMaskWrapper     (mask invalid actions)
        -> DenseRewardWrapper  (reward shaping)
          -> [StealthAwareWrapper if stealth mode]
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class IntActionWrapper(gym.Wrapper):
    """Coerce actions to Python int -- required by NASim >= 0.10.

    SB3 models return numpy integer types (np.int64).  NASim's flat-action
    step() requires a plain Python int and raises an error otherwise.
    This thin wrapper sits directly on top of the raw NASim env and
    converts the action type before forwarding.
    """

    def step(self, action):
        return self.env.step(int(action))


class ActionMaskWrapper(gym.Wrapper):
    """Expose NASim's action validity mask for use with sb3-contrib MaskablePPO.

    NASim 0.12 has a ``get_action_mask()`` method but it contains a bug
    (calls ``host_discovered`` on the Network object instead of the State
    object).  This wrapper fixes the mask computation and exposes it via
    the ``action_masks()`` method that MaskablePPO expects.

    An action is valid when the target host has been both discovered and
    is reachable from the current foothold. This reduces the effective
    action space from 110 to ~5-15 valid actions per step, making
    exploration tractable.
    """

    def _is_action_valid(self, action_idx: int) -> bool:
        """Return whether an action is valid in the current NASim state.

        An action is valid when its target host has been:
          (a) discovered -- the agent has previously scanned the subnet and
              knows the host exists, AND
          (b) reachable  -- there is a network path from a compromised host
              to the target (respecting firewall rules).

        This drastically reduces the effective action space from 110 to
        roughly 5-15 valid actions per step, making exploration tractable.
        """
        nasim_env = self.unwrapped
        state = getattr(nasim_env, "current_state", None)
        if state is None:
            return True

        action = nasim_env.action_space.get_action(action_idx)
        target = getattr(action, "target", None)
        if target is None:
            return True

        return bool(state.host_discovered(target) and state.host_reachable(target))

    def action_masks(self) -> np.ndarray:
        """Build a binary mask over the full action space.

        Returns a float32 array of shape (n_actions,) where 1.0 = valid
        and 0.0 = invalid.  MaskablePPO uses this to zero out logits
        before softmax; DQN uses it to set Q-values to -inf before argmax.
        """
        nasim_env = self.unwrapped
        n_actions = nasim_env.action_space.n
        mask = np.zeros(n_actions, dtype=np.float32)

        for a_idx in range(n_actions):
            if self._is_action_valid(a_idx):
                mask[a_idx] = 1.0

        # Defensive fallback: if no actions are valid (shouldn't happen in
        # a well-configured NASim scenario), allow everything to prevent
        # a crash in the agent's action selection.
        if mask.sum() == 0:
            mask[:] = 1.0
        return mask

    def step(self, action: int):
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

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        dense_bonus = 0.0
        n_changed = 0

        if self._prev_obs is not None:
            n_changed = int(
                np.sum(np.abs(obs.astype(np.float32) - self._prev_obs.astype(np.float32)) > 1e-6)
            )
            if n_changed > 1:
                dense_bonus = float(self.progress_bonus)
                reward += dense_bonus

        self._prev_obs = obs.copy()
        info["dense_progress_bonus"] = dense_bonus
        info["dense_n_features_changed"] = n_changed
        return obs, reward, terminated, truncated, info
