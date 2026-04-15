"""
ppo_agent.py
------------
Proximal Policy Optimization (PPO) agent for NASim attack-path simulation.

Author: Syed Ali Turab
Course: MMAI 845 -- Reinforcement Learning

Algorithm overview
==================
PPO is an *on-policy*, *policy gradient* method.  It directly parameterises
a stochastic policy pi(a|s) and optimises it by collecting rollouts of
experience, computing advantages using Generalised Advantage Estimation (GAE),
and performing clipped gradient updates that prevent destructively large
policy changes.

Why PPO for this problem?
=========================
1. **Action masking**: sb3-contrib provides MaskablePPO, which natively zeros
   out logits for invalid actions before the softmax.  This means every action
   sampled during training is guaranteed valid -- critical when 70-80% of
   NASim's 110 actions are invalid at any given state.
2. **Stability**: PPO's clipped objective and on-policy rollouts provide
   stable learning, which is important for sparse-reward environments where
   a single bad update can erase progress.
3. **Entropy regularisation**: the entropy coefficient (ent_coef) encourages
   exploration by penalising overly deterministic policies.  We set it higher
   than default (0.05 vs 0.01) because the large invalid-action space requires
   more exploration of valid alternatives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


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


class MaskedEvalCallback(BaseCallback):
    """Evaluate MaskablePPO with action masks at regular intervals.

    SB3's built-in EvalCallback doesn't pass action_masks during predict,
    so MaskablePPO evaluation always selects from the full (unmasked)
    action space, producing garbage results.  This callback manually
    rolls out episodes with proper masking.
    """

    def __init__(
        self,
        eval_env: gym.Env,
        n_eval_episodes: int = 10,
        eval_freq: int = 10_000,
        best_model_save_path: str | None = None,
        deterministic: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        rewards = []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total = 0.0
            while not done:
                mask = (
                    self.eval_env.action_masks() if hasattr(self.eval_env, "action_masks") else None
                )
                action, _ = self.model.predict(
                    obs, deterministic=self.deterministic, action_masks=mask
                )
                obs, r, term, trunc, _ = self.eval_env.step(int(action))
                done = term or trunc
                total += float(r)
            rewards.append(total)

        mean_r = float(np.mean(rewards))
        self.logger.record("eval/mean_reward", mean_r)
        self.logger.record("eval/std_reward", float(np.std(rewards)))

        if self.best_model_save_path and mean_r > self.best_mean_reward:
            self.best_mean_reward = mean_r
            self.model.save(f"{self.best_model_save_path}/best_model")

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
            eval_cb = MaskedEvalCallback(
                eval_env=eval_env,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_freq,
                best_model_save_path=save_path,
                deterministic=False,
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

    def predict(
        self, obs: np.ndarray, deterministic: bool = False, action_masks: np.ndarray | None = None
    ) -> int:
        """Return an action for a given observation."""
        action, _ = self.model.predict(obs, deterministic=deterministic, action_masks=action_masks)
        return int(action)

    def run_episode(
        self,
        env: gym.Env,
        deterministic: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        Roll out one full episode and return metrics.

        Uses stochastic policy by default because the deterministic policy
        can get stuck repeating a single action.
        """
        obs, info = env.reset(seed=seed) if seed is not None else env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        path: list[int] = []
        target_path: list[str] = []

        while not done:
            mask = env.action_masks() if hasattr(env, "action_masks") else None
            action = self.predict(obs, deterministic=deterministic, action_masks=mask)
            path.append(action)
            target = None
            if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "action_space"):
                nasim_action = env.unwrapped.action_space.get_action(int(action))
                target = getattr(nasim_action, "target", None)
            target_path.append(str(target) if target is not None else "None")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        return {
            "total_reward": total_reward,
            "steps": steps,
            "path": path,
            "target_path": target_path,
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
