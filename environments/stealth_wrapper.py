"""
stealth_wrapper.py
------------------
Gymnasium-compatible wrapper that adds stealth-aware reward shaping on top
of any NASim environment.

Author: Team Broadview
Course: MMAI 845 -- Reinforcement Learning

Motivation
==========
Real attackers must balance speed against detection.  Every scan, exploit,
or privilege escalation generates network traffic and log entries that a
Security Operations Centre (SOC) or Intrusion Detection System (IDS) may
flag.  The stealth wrapper models this trade-off by:

  1. Accumulating a "detection score" on every active (non-idle) step.
  2. Subtracting a penalty from the reward proportional to detection risk.
  3. Terminating the episode (attacker caught) when cumulative detection
     exceeds a threshold.

This encourages the RL agent to learn *efficient* attack paths that
minimise unnecessary actions, rather than brute-forcing every host.

Reward shaping
==============
On **active** steps (any action that changes state or costs resources)::

    r' = r_env  -  alpha * detection_cost_per_step

On **idle** steps (action 0, noop) the reward passes through unchanged
and detection does not accumulate.

When cumulative detection >= detection_threshold, the episode terminates
with an additional caught_penalty (-100 by default).

With default parameters (threshold=0.8, cost=0.1, alpha=1.0), the agent
has at most 8 active steps before being caught.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class StealthAwareWrapper(gym.Wrapper):
    """
    Stealth-aware reward wrapper for NASim environments.

    Parameters
    ----------
    env : gym.Env
        A NASim Gymnasium environment.
    detection_threshold : float
        Cumulative detection score at which the episode is terminated
        (attacker caught). Default: 0.8.
    detection_cost_per_step : float
        Amount added to the cumulative detection score on every non-idle step.
        Default: 0.1.
    caught_penalty : float
        Reward penalty applied at the terminal step when the attacker is caught.
        Default: -100.0.
    alpha : float
        Coefficient for subtracting the per-step detection cost from the
        environment reward. Default: 1.0.
    """

    def __init__(
        self,
        env: gym.Env,
        detection_threshold: float = 0.8,
        detection_cost_per_step: float = 0.1,
        caught_penalty: float = -100.0,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(env)
        self.detection_threshold = detection_threshold
        self.detection_cost_per_step = detection_cost_per_step
        self.caught_penalty = caught_penalty
        self.alpha = alpha

        # Tracked state
        self._cumulative_detection: float = 0.0
        self._steps: int = 0
        self._episode_rewards: list[float] = []

    # ------------------------------------------------------------------
    # Core Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        self._cumulative_detection = 0.0
        self._steps = 0
        self._episode_rewards = []
        obs, info = self.env.reset(**kwargs)
        info["cumulative_detection"] = self._cumulative_detection
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        base_reward = float(reward)
        detection_penalty = 0.0

        self._steps += 1

        # An action is "active" (generates detection risk) when the agent
        # performs an exploit, scan, or priv-esc — not when it idles.
        # NASim uses action 0 as the noop and returns reward == 0 for it.
        is_active = reward != 0 or action != 0

        if is_active:
            self._cumulative_detection += self.detection_cost_per_step
            detection_penalty = float(self.alpha * self.detection_cost_per_step)
            shaped_reward = reward - detection_penalty
        else:
            shaped_reward = reward

        self._episode_rewards.append(shaped_reward)

        # Check if attacker was caught
        caught = self._cumulative_detection >= self.detection_threshold
        if caught and not terminated:
            shaped_reward += self.caught_penalty
            terminated = True

        info["cumulative_detection"] = self._cumulative_detection
        info["detection_threshold"] = self.detection_threshold
        info["caught"] = caught
        info["episode_steps"] = self._steps
        info["base_reward"] = base_reward
        info["detection_penalty"] = detection_penalty

        return obs, shaped_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def cumulative_detection(self) -> float:
        """Current cumulative detection score for the running episode."""
        return self._cumulative_detection

    @property
    def stealth_budget_remaining(self) -> float:
        """Fraction of detection budget remaining (1.0 = full, 0.0 = caught)."""
        return max(0.0, 1.0 - self._cumulative_detection / self.detection_threshold)

    def detection_summary(self) -> dict:
        """Return a dict summary of detection metrics for the last episode."""
        return {
            "cumulative_detection": self._cumulative_detection,
            "detection_threshold": self.detection_threshold,
            "steps": self._steps,
            "total_shaped_reward": sum(self._episode_rewards),
            "caught": self._cumulative_detection >= self.detection_threshold,
        }


def make_stealth_env(
    base_env: gym.Env,
    detection_threshold: float = 0.8,
    detection_cost_per_step: float = 0.1,
    caught_penalty: float = -100.0,
    alpha: float = 1.0,
) -> StealthAwareWrapper:
    """
    Convenience function to wrap a NASim env with stealth-aware reward shaping.

    Parameters
    ----------
    base_env : gym.Env
        Any NASim environment returned by nasim.make() or nasim.load().

    Returns
    -------
    StealthAwareWrapper
    """
    return StealthAwareWrapper(
        env=base_env,
        detection_threshold=detection_threshold,
        detection_cost_per_step=detection_cost_per_step,
        caught_penalty=caught_penalty,
        alpha=alpha,
    )
