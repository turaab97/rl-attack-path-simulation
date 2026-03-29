"""
ppo_agent.py
------------
Proximal Policy Optimization (PPO) agent for NASim attack-path simulation.

Author: Syed Ali Turab
Course: MMAI 845 – Reinforcement Learning

Uses sb3-contrib MaskablePPO with action masking.  NASim has 110 actions but
only ~30 are valid at any time.  Without masking the agent wastes most of its
exploration budget on invalid actions that always return -1.  MaskablePPO
zeroes out invalid action logits before sampling, so every training step
interacts meaningfully with the environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.wrappers import ActionMaskWrapper


class EpisodeStatsCallback(BaseCallback):
    """Log per-episode stats to TensorBoard."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                self.logger.record("train/ep_rew_mean", ep["r"])
                self.logger.record("train/ep_len_mean", ep["l"])
                if "cumulative_detection" in info:
                    self.logger.record("train/cumulative_detection", info["cumulative_detection"])
        return True


class PPOAttackAgent:
    """
    MaskablePPO-based attacker agent wrapping sb3-contrib.

    Parameters
    ----------
    env : gym.Env
        A NASim environment wrapped with ActionMaskWrapper.
    learning_rate : float
        Adam learning rate. Default: 3e-4.
    n_steps : int
        Steps collected per rollout before each PPO update. Default: 2048.
    batch_size : int
        Mini-batch size for each PPO epoch. Default: 64.
    n_epochs : int
        Number of optimisation epochs per rollout. Default: 10.
    gamma : float
        Discount factor. Default: 0.99.
    gae_lambda : float
        GAE lambda for advantage estimation. Default: 0.95.
    clip_range : float
        PPO clipping parameter. Default: 0.2.
    ent_coef : float
        Entropy regularisation coefficient. Default: 0.05.
    tensorboard_log : str or None
        Directory for TensorBoard logs. Default: 'runs/ppo'.
    policy_kwargs : dict or None
        Extra keyword arguments forwarded to MlpPolicy.
    seed : int
        Random seed. Default: 42.
    device : str
        Torch device. Default: 'cpu'.
    """

    NAME = "PPO"

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.05,
        tensorboard_log: str | None = "runs/ppo",
        policy_kwargs: dict | None = None,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self.env = Monitor(env)
        self.seed = seed

        if policy_kwargs is None:
            policy_kwargs = dict(net_arch=[256, 256])

        self.model = MaskablePPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            verbose=0,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int = 500_000,
        eval_env: gym.Env | None = None,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        save_path: str | None = None,
        reward_threshold: float | None = None,
        tb_log_name: str = "ppo_run",
    ) -> MaskablePPO:
        """Train the MaskablePPO agent."""
        callbacks = [EpisodeStatsCallback()]

        if eval_env is not None:
            eval_monitor = Monitor(eval_env)
            eval_cb = EvalCallback(
                eval_monitor,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                verbose=0,
            )
            callbacks.append(eval_cb)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=tb_log_name,
            progress_bar=True,
        )
        return self.model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, deterministic: bool = False, action_masks: np.ndarray | None = None) -> int:
        """Return an action for a given observation."""
        action, _ = self.model.predict(obs, deterministic=deterministic, action_masks=action_masks)
        return int(action)

    def run_episode(self, env: gym.Env, deterministic: bool = False) -> dict[str, Any]:
        """
        Roll out one full episode and return metrics.

        Uses stochastic policy by default because the deterministic policy
        can get stuck repeating a single action.
        """
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        path: list[int] = []

        while not done:
            mask = env.action_masks() if hasattr(env, "action_masks") else None
            action = self.predict(obs, deterministic=deterministic, action_masks=mask)
            path.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        return {
            "total_reward": total_reward,
            "steps": steps,
            "path": path,
            "caught": info.get("caught", False),
            "cumulative_detection": info.get("cumulative_detection", None),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model to `path` (SB3 adds .zip automatically)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        print(f"[PPO] Model saved → {path}.zip")

    @classmethod
    def load(cls, path: str, env: gym.Env) -> "PPOAttackAgent":
        """Load a previously saved model."""
        agent = cls.__new__(cls)
        agent.env = Monitor(env)
        agent.model = MaskablePPO.load(path, env=agent.env)
        print(f"[PPO] Model loaded ← {path}")
        return agent
