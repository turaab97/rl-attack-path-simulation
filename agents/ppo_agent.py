"""
ppo_agent.py
------------
Proximal Policy Optimization (PPO) agent for NASim attack-path simulation.

Author: Syed Ali Turab
Course: MMAI 845 – Reinforcement Learning

Uses Stable-Baselines3 PPO with an MLP policy.  NASim exposes a flat
observation vector and a discrete action space, so MlpPolicy is the natural
choice.

Key design choices
==================
* The agent is wrapped in a thin class so callers can swap PPO ↔ DQN without
  changing the training / evaluation harness.
* TensorBoard logging is enabled by default; pass tensorboard_log=None to
  disable.
* The model is saved/loaded via the SB3 native API (.zip format).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.wrappers import IntActionWrapper


class EpisodeStatsCallback(BaseCallback):
    """
    Logs per-episode statistics (reward, length, detection score) to
    TensorBoard so we can track attacker behaviour over training.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []

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
    PPO-based attacker agent wrapping Stable-Baselines3.

    Parameters
    ----------
    env : gym.Env
        The (optionally stealth-wrapped) NASim environment.
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
        PPO clipping parameter ε. Default: 0.2.
    ent_coef : float
        Entropy regularisation coefficient (encourages exploration). Default: 0.01.
    tensorboard_log : str or None
        Directory for TensorBoard logs. Pass None to disable. Default: 'runs/ppo'.
    policy_kwargs : dict or None
        Extra keyword arguments forwarded to MlpPolicy (e.g. net_arch).
    seed : int
        Random seed. Default: 42.
    device : str
        Torch device: 'auto', 'cpu', or 'cuda'. Default: 'auto'.
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
        ent_coef: float = 0.01,
        tensorboard_log: str | None = "runs/ppo",
        policy_kwargs: dict | None = None,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        self.env = Monitor(IntActionWrapper(env))
        self.seed = seed

        if policy_kwargs is None:
            # Two hidden layers, 256 units each – appropriate for NASim obs size
            policy_kwargs = dict(net_arch=[256, 256])

        self.model = PPO(
            policy="MlpPolicy",
            env=DummyVecEnv([lambda: self.env]),
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
    ) -> PPO:
        """
        Train the PPO agent.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps to train for.
        eval_env : gym.Env or None
            Optional held-out environment for periodic evaluation.
        eval_freq : int
            Evaluate every `eval_freq` steps (only used if eval_env is given).
        n_eval_episodes : int
            Number of episodes per evaluation.
        save_path : str or None
            If provided, saves the best model checkpoint here.
        reward_threshold : float or None
            Stop training early once mean reward exceeds this value.
        tb_log_name : str
            Sub-directory name inside tensorboard_log.

        Returns
        -------
        stable_baselines3.PPO
            The trained model.
        """
        callbacks = [EpisodeStatsCallback()]

        if eval_env is not None:
            eval_monitor = Monitor(eval_env)
            callback_list = []
            if reward_threshold is not None:
                stop_cb = StopTrainingOnRewardThreshold(
                    reward_threshold=reward_threshold, verbose=1
                )
                callback_list.append(stop_cb)
            eval_cb = EvalCallback(
                DummyVecEnv([lambda: eval_monitor]),
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                callback_on_new_best=callback_list[0] if callback_list else None,
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

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Return the greedy (or stochastic) action for a given observation."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def run_episode(self, env: gym.Env, deterministic: bool = True) -> dict[str, Any]:
        """
        Roll out one full episode and return metrics.

        Returns
        -------
        dict with keys: total_reward, steps, path, caught, cumulative_detection
        """
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        path: list[int] = []

        while not done:
            action = self.predict(obs, deterministic=deterministic)
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
        agent.model = PPO.load(path, env=DummyVecEnv([lambda: agent.env]))
        print(f"[PPO] Model loaded ← {path}")
        return agent
