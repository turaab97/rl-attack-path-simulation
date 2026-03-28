"""
wrappers.py
-----------
Shared Gymnasium wrappers used by both PPO and DQN agents.

Author: Syed Ali Turab
Course: MMAI 845 – Reinforcement Learning
"""

from __future__ import annotations

import gymnasium as gym


class IntActionWrapper(gym.Wrapper):
    """Coerce actions to Python int — required by nasim >= 0.10 with NumPy 2.x.

    NASim's step() expects a plain Python int, but SB3 may pass numpy
    integer types that NASim rejects.  This wrapper sits between the
    SB3 model and the environment to ensure type compatibility.
    """

    def step(self, action):
        return self.env.step(int(action))
