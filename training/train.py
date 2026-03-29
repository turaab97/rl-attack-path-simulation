"""
train.py
--------
Main training entry point for RL attack-path simulation.

Author: Syed Ali Turab
Course: MMAI 845 – Reinforcement Learning

Usage (CLI)
===========
python -m training.train --agent ppo --stealth --timesteps 500000
python -m training.train --agent dqn --timesteps 300000
python -m training.train --compare --timesteps 200000  # runs both, saves results

All artefacts (models, logs, results) go under the ``results/`` directory
relative to the working directory (or the path set in ``--output_dir``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agents.dqn_agent import DQNAttackAgent
from agents.ppo_agent import PPOAttackAgent
from agents.wrappers import DenseRewardWrapper, IntActionWrapper
from environments.network_config import make_env
from environments.stealth_wrapper import make_stealth_env

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_environments(
    scenario_name: str | None,
    stealth: bool,
    detection_threshold: float,
    detection_cost_per_step: float,
    caught_penalty: float,
    alpha: float,
) -> tuple:
    """Return (train_env, eval_env).

    The training environment is wrapped with DenseRewardWrapper to provide
    intermediate reward shaping (observation-diff bonus), enabling the
    agent to learn from incremental progress rather than only from the
    extremely sparse sensitive-host rewards.

    The eval environment uses the true NASim rewards (no shaping) so that
    evaluation metrics reflect genuine task performance.

    Both environments use IntActionWrapper so NASim receives plain ints.
    """
    train_env = DenseRewardWrapper(IntActionWrapper(make_env(scenario_name=scenario_name)))
    eval_env = IntActionWrapper(make_env(scenario_name=scenario_name))

    if stealth:
        train_env = make_stealth_env(
            train_env,
            detection_threshold=detection_threshold,
            detection_cost_per_step=detection_cost_per_step,
            caught_penalty=caught_penalty,
            alpha=alpha,
        )
        eval_env = make_stealth_env(
            eval_env,
            detection_threshold=detection_threshold,
            detection_cost_per_step=detection_cost_per_step,
            caught_penalty=caught_penalty,
            alpha=alpha,
        )
    return train_env, eval_env


# ---------------------------------------------------------------------------
# Single-agent training
# ---------------------------------------------------------------------------


def train_agent(
    agent_type: str = "ppo",
    scenario_name: str | None = None,
    stealth: bool = False,
    total_timesteps: int = 500_000,
    output_dir: str = "results",
    detection_threshold: float = 0.8,
    detection_cost_per_step: float = 0.1,
    caught_penalty: float = -100.0,
    alpha: float = 1.0,
    seed: int = 42,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 10,
) -> dict:
    """
    Train a single agent and return a results dict.

    Parameters
    ----------
    agent_type : str
        'ppo' or 'dqn'.
    scenario_name : str or None
        NASim built-in scenario name, or None for the custom AI-infra topology.
    stealth : bool
        Whether to apply the stealth-aware reward wrapper.
    total_timesteps : int
        Training budget in environment steps.
    output_dir : str
        Root directory for saving models and logs.
    detection_threshold : float
        Cumulative detection threshold (stealth mode only).
    detection_cost_per_step : float
        Detection cost per active step (stealth mode only).
    caught_penalty : float
        Penalty when agent is caught (stealth mode only).
    alpha : float
        Reward penalty coefficient (stealth mode only).
    seed : int
        Random seed for reproducibility.
    eval_freq : int
        Evaluate every N timesteps.
    n_eval_episodes : int
        Episodes per evaluation round.

    Returns
    -------
    dict
        Training metadata including full hyperparameters.
    """
    tag = f"{agent_type}_{'stealth' if stealth else 'baseline'}"
    save_dir = Path(output_dir) / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training {agent_type.upper()} | stealth={stealth} | steps={total_timesteps:,}")
    print(f"  Saving to: {save_dir}")
    print(f"{'='*60}\n")

    train_env, eval_env = _make_environments(
        scenario_name=scenario_name,
        stealth=stealth,
        detection_threshold=detection_threshold,
        detection_cost_per_step=detection_cost_per_step,
        caught_penalty=caught_penalty,
        alpha=alpha,
    )

    tb_dir = save_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_log = str(tb_dir)

    if agent_type.lower() == "ppo":
        agent = PPOAttackAgent(
            env=train_env,
            tensorboard_log=tb_log,
            seed=seed,
        )
        tb_log_name = f"ppo_{'stealth' if stealth else 'baseline'}"
    elif agent_type.lower() == "dqn":
        agent = DQNAttackAgent(
            env=train_env,
            tensorboard_log=tb_log,
            seed=seed,
        )
        tb_log_name = f"dqn_{'stealth' if stealth else 'baseline'}"
    else:
        raise ValueError(f"Unknown agent type: {agent_type!r}. Choose 'ppo' or 'dqn'.")

    t0 = time.time()
    agent.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        save_path=str(save_dir / "best_model"),
        tb_log_name=tb_log_name,
    )
    wall_time = time.time() - t0

    # Save final model
    model_path = str(save_dir / "final_model")
    agent.save(model_path)

    # Persist full metadata including hyperparameters for reproducibility
    agent_hparams = {
        k: v
        for k, v in agent.model.__dict__.items()
        if k
        in {
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "buffer_size",
            "learning_starts",
            "tau",
            "train_freq",
            "gradient_steps",
            "target_update_interval",
            "exploration_fraction",
            "exploration_initial_eps",
            "exploration_final_eps",
        }
    }
    # SB3 stores some values as schedules/enums; convert for JSON
    for k, v in agent_hparams.items():
        if callable(v):
            try:
                agent_hparams[k] = float(v(1.0))
            except Exception:
                agent_hparams[k] = str(v)
        elif hasattr(v, "value"):
            agent_hparams[k] = v.value
        elif not isinstance(v, (int, float, str, bool, list, type(None))):
            agent_hparams[k] = str(v)

    meta = {
        "agent": agent_type,
        "scenario": scenario_name or "custom_ai_infra",
        "stealth": stealth,
        "total_timesteps": total_timesteps,
        "wall_time_seconds": round(wall_time, 2),
        "save_dir": str(save_dir),
        "model_path": model_path + ".zip",
        "seed": seed,
        "eval_freq": eval_freq,
        "n_eval_episodes": n_eval_episodes,
        "stealth_params": (
            {
                "detection_threshold": detection_threshold,
                "detection_cost_per_step": detection_cost_per_step,
                "caught_penalty": caught_penalty,
                "alpha": alpha,
            }
            if stealth
            else None
        ),
        "hyperparameters": agent_hparams,
        "net_arch": [256, 256],
    }
    with open(save_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[{agent_type.upper()}] Training complete in {wall_time:.1f}s")
    train_env.close()
    eval_env.close()
    return meta


# ---------------------------------------------------------------------------
# Head-to-head comparison
# ---------------------------------------------------------------------------


def run_comparison(
    scenario_name: str | None = None,
    stealth: bool = False,
    total_timesteps: int = 300_000,
    output_dir: str = "results",
    detection_threshold: float = 0.8,
    detection_cost_per_step: float = 0.1,
    caught_penalty: float = -100.0,
    alpha: float = 1.0,
    seed: int = 42,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 10,
) -> dict:
    """
    Train both PPO and DQN under the same conditions and save a comparison
    summary to ``output_dir/comparison_summary.json``.

    Returns
    -------
    dict
        {'ppo': meta_dict, 'dqn': meta_dict}
    """
    shared_kwargs = dict(
        scenario_name=scenario_name,
        stealth=stealth,
        total_timesteps=total_timesteps,
        output_dir=output_dir,
        detection_threshold=detection_threshold,
        detection_cost_per_step=detection_cost_per_step,
        caught_penalty=caught_penalty,
        alpha=alpha,
        seed=seed,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
    )

    ppo_meta = train_agent(agent_type="ppo", **shared_kwargs)
    dqn_meta = train_agent(agent_type="dqn", **shared_kwargs)

    comparison = {"ppo": ppo_meta, "dqn": dqn_meta}
    out_path = Path(output_dir) / "comparison_summary.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison summary saved → {out_path}")
    return comparison


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO / DQN attack-path agents on NASim.")
    p.add_argument(
        "--agent",
        choices=["ppo", "dqn"],
        default="ppo",
        help="Agent algorithm (default: ppo)",
    )
    p.add_argument(
        "--scenario",
        default=None,
        help="NASim built-in scenario name (e.g. 'small-linear'). "
        "Omit to use the custom AI-infra topology.",
    )
    p.add_argument(
        "--stealth",
        action="store_true",
        help="Apply stealth-aware reward wrapper.",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500000).",
    )
    p.add_argument(
        "--output_dir",
        default="results",
        help="Root directory for output artefacts (default: results/).",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Train both PPO and DQN and save a comparison summary.",
    )
    p.add_argument(
        "--detection_threshold",
        type=float,
        default=0.8,
        help="Detection threshold for stealth wrapper (default: 0.8).",
    )
    p.add_argument(
        "--detection_cost",
        type=float,
        default=0.1,
        help="Detection cost per active step (default: 0.1).",
    )
    p.add_argument(
        "--caught_penalty",
        type=float,
        default=-100.0,
        help="Penalty when attacker is caught (default: -100.0).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Stealth penalty coefficient (default: 1.0).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--eval_freq",
        type=int,
        default=10_000,
        help="Evaluate every N timesteps (default: 10000).",
    )
    p.add_argument(
        "--n_eval_episodes",
        type=int,
        default=10,
        help="Episodes per evaluation round (default: 10).",
    )
    return p.parse_args()


def main() -> None:
    """CLI entry point (also registered as ``train-agent`` console script)."""
    args = _parse_args()

    common_kwargs = dict(
        scenario_name=args.scenario,
        stealth=args.stealth,
        total_timesteps=args.timesteps,
        output_dir=args.output_dir,
        detection_threshold=args.detection_threshold,
        detection_cost_per_step=args.detection_cost,
        caught_penalty=args.caught_penalty,
        alpha=args.alpha,
        seed=args.seed,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
    )

    if args.compare:
        run_comparison(**common_kwargs)
    else:
        train_agent(agent_type=args.agent, **common_kwargs)


if __name__ == "__main__":
    main()
