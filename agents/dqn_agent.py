"""
dqn_agent.py
------------
Deep Q-Network (DQN) agent for NASim attack-path simulation.

Uses Stable-Baselines3 DQN with an MLP policy.  DQN is particularly useful
here because NASim's discrete action space maps naturally to a Q-table
approximated by an MLP; epsilon-greedy exploration helps the agent discover
novel attack paths.

Mirrors the PPOAttackAgent interface so both agents can be used
interchangeably by the training harness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


class EpisodeStatsCallback(BaseCallback):
    """Log per-episode stats to TensorBoard."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                self.logger.record("train/ep_rew_mean", ep["r"])
                self.logger.record("train/ep_len_mean", ep["l"])
                if "cumulative_detection" in info:
                    self.logger.record(
                        "train/cumulative_detection", info["cumulative_detection"]
                    )
        return True


class DQNAttackAgent:
    """
    DQN-based attacker agent wrapping Stable-Baselines3.

    Parameters
    ----------
    env : gym.Env
        The (optionally stealth-wrapped) NASim environment.
    learning_rate : float
        Adam learning rate. Default: 1e-4.
    buffer_size : int
        Replay buffer capacity. Default: 100_000.
    learning_starts : int
        Steps before first gradient update. Default: 1_000.
    batch_size : int
        Batch size sampled from the replay buffer. Default: 64.
    tau : float
        Soft-update coefficient for target network. Default: 1.0 (hard update).
    gamma : float
        Discount factor. Default: 0.99.
    train_freq : int
        Update the model every `train_freq` steps. Default: 4.
    gradient_steps : int
        Gradient steps per update. Default: 1.
    target_update_interval : int
        Steps between target network updates. Default: 1000.
    exploration_fraction : float
        Fraction of total training during which ε decays. Default: 0.2.
    exploration_initial_eps : float
        Starting ε for ε-greedy exploration. Default: 1.0.
    exploration_final_eps : float
        Final ε after decay. Default: 0.05.
    tensorboard_log : str or None
        Directory for TensorBoard logs. Default: 'runs/dqn'.
    policy_kwargs : dict or None
        Extra kwargs for MlpPolicy (e.g. net_arch).
    seed : int
        Random seed. Default: 42.
    device : str
        Torch device. Default: 'auto'.
    """

    NAME = "DQN"

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 1e-4,
        buffer_size: int = 100_000,
        learning_starts: int = 1_000,
        batch_size: int = 64,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 1_000,
        exploration_fraction: float = 0.2,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        tensorboard_log: str | None = "runs/dqn",
        policy_kwargs: dict | None = None,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        self.env = Monitor(env)
        self.seed = seed

        if policy_kwargs is None:
            policy_kwargs = dict(net_arch=[256, 256])

        self.model = DQN(
            policy="MlpPolicy",
            env=DummyVecEnv([lambda: self.env]),
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
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
        tb_log_name: str = "dqn_run",
    ) -> DQN:
        """
        Train the DQN agent.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps.
        eval_env : gym.Env or None
            Optional held-out environment for periodic evaluation.
        eval_freq : int
            Evaluate every `eval_freq` steps.
        n_eval_episodes : int
            Episodes per evaluation.
        save_path : str or None
            Directory to save the best checkpoint.
        reward_threshold : float or None
            Stop early once mean eval reward exceeds this value.
        tb_log_name : str
            Sub-directory name inside tensorboard_log.

        Returns
        -------
        stable_baselines3.DQN
        """
        callbacks = [EpisodeStatsCallback()]

        if eval_env is not None:
            eval_monitor = Monitor(eval_env)
            callback_on_best = None
            if reward_threshold is not None:
                callback_on_best = StopTrainingOnRewardThreshold(
                    reward_threshold=reward_threshold, verbose=1
                )
            eval_cb = EvalCallback(
                DummyVecEnv([lambda: eval_monitor]),
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                callback_on_new_best=callback_on_best,
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
        """Return the greedy (or ε-greedy) action for a given observation."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def run_episode(
        self, env: gym.Env, deterministic: bool = True
    ) -> dict[str, Any]:
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
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        print(f"[DQN] Model saved → {path}.zip")

    @classmethod
    def load(cls, path: str, env: gym.Env) -> "DQNAttackAgent":
        agent = cls.__new__(cls)
        agent.env = Monitor(env)
        agent.model = DQN.load(path, env=DummyVecEnv([lambda: agent.env]))
        print(f"[DQN] Model loaded ← {path}")
        return agent
