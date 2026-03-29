"""
dqn_agent.py
------------
Deep Q-Network (DQN) agent for NASim attack-path simulation.

Author: Syed Ali Turab
Course: MMAI 845 – Reinforcement Learning

Uses Stable-Baselines3 DQN with manual action masking during exploration.
DQN's epsilon-greedy exploration is modified so that random actions are
sampled only from the set of valid actions (via ActionMaskWrapper), and
greedy action selection masks invalid Q-values with -inf.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.wrappers import ActionMaskWrapper


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
                    self.logger.record("train/cumulative_detection", info["cumulative_detection"])
        return True


class MaskedEvalCallback(BaseCallback):
    """Evaluate DQN with masked Q-values at regular intervals."""

    def __init__(
        self,
        eval_env: gym.Env,
        agent: "DQNAttackAgent",
        n_eval_episodes: int = 10,
        eval_freq: int = 10_000,
        best_model_save_path: str | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.agent = agent
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        rewards = []
        for _ in range(self.n_eval_episodes):
            result = self.agent.run_episode(self.eval_env, deterministic=True)
            rewards.append(result["total_reward"])

        mean_r = float(np.mean(rewards))
        self.logger.record("eval/mean_reward", mean_r)
        self.logger.record("eval/std_reward", float(np.std(rewards)))

        if self.best_model_save_path and mean_r > self.best_mean_reward:
            self.best_mean_reward = mean_r
            self.model.save(f"{self.best_model_save_path}/best_model")

        return True


class DQNAttackAgent:
    """
    DQN-based attacker agent with manual action masking.

    Parameters
    ----------
    env : gym.Env
        A NASim environment wrapped with ActionMaskWrapper.
    learning_rate : float
        Adam learning rate. Default: 1e-4.
    buffer_size : int
        Replay buffer capacity. Default: 100_000.
    learning_starts : int
        Steps before first gradient update. Default: 1_000.
    batch_size : int
        Batch size sampled from the replay buffer. Default: 64.
    tau : float
        Soft-update coefficient for target network. Default: 1.0.
    gamma : float
        Discount factor. Default: 0.99.
    train_freq : int
        Update the model every N steps. Default: 4.
    gradient_steps : int
        Gradient steps per update. Default: 1.
    target_update_interval : int
        Steps between target network updates. Default: 1000.
    exploration_fraction : float
        Fraction of total training during which epsilon decays. Default: 0.5.
    exploration_initial_eps : float
        Starting epsilon. Default: 1.0.
    exploration_final_eps : float
        Final epsilon. Default: 0.05.
    tensorboard_log : str or None
        TensorBoard log directory. Default: 'runs/dqn'.
    policy_kwargs : dict or None
        Extra kwargs for MlpPolicy.
    seed : int
        Random seed. Default: 42.
    device : str
        Torch device. Default: 'cpu'.
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
        exploration_fraction: float = 0.5,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        tensorboard_log: str | None = "runs/dqn",
        policy_kwargs: dict | None = None,
        seed: int = 42,
        device: str = "cpu",
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
        """Train the DQN agent."""
        callbacks = [EpisodeStatsCallback()]

        if eval_env is not None:
            eval_cb = MaskedEvalCallback(
                eval_env=eval_env,
                agent=self,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_freq,
                best_model_save_path=save_path,
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

    def predict(self, obs: np.ndarray, deterministic: bool = True, action_masks: np.ndarray | None = None) -> int:
        """Return the action for a given observation with optional masking."""
        if action_masks is not None and deterministic:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.model.device)
            with torch.no_grad():
                q_values = self.model.q_net(obs_tensor).squeeze(0).cpu().numpy()
            mask_np = np.array(action_masks, dtype=np.float32)
            q_values[mask_np < 0.5] = -np.inf
            return int(np.argmax(q_values))
        if action_masks is not None and not deterministic:
            valid = np.where(np.array(action_masks) > 0.5)[0]
            if len(valid) > 0 and np.random.random() < getattr(self.model, "exploration_rate", 0.05):
                return int(np.random.choice(valid))
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def run_episode(self, env: gym.Env, deterministic: bool = True) -> dict[str, Any]:
        """Roll out one full episode with masked action selection."""
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
