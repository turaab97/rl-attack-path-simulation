"""
evaluate.py
-----------
Evaluation harness for trained PPO / DQN attack-path agents.

Author: Syed Ali Turab
Course: MMAI 845 – Reinforcement Learning

Usage (CLI)
===========
python -m training.evaluate \\
    --ppo_model results/ppo_baseline/final_model \\
    --dqn_model results/dqn_baseline/final_model \\
    --episodes 100 --stealth

Outputs
=======
* Console summary table
* results/eval_results.json  (per-episode and aggregate metrics)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from agents.dqn_agent import DQNAttackAgent
from agents.ppo_agent import PPOAttackAgent
from agents.wrappers import ActionMaskWrapper, IntActionWrapper
from environments.network_config import make_env
from environments.stealth_wrapper import make_stealth_env

# Reward thresholds for sensitive hosts — used to determine whether the
# agent actually compromised AI infrastructure vs just wandering around.
_SENSITIVE_HOST_TOTAL_VALUE = sum(
    v
    for _, v in [
        ((3, 0), 200),
        ((3, 1), 200),
        ((3, 2), 150),
        ((4, 0), 300),
    ]
)  # = 850

# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def evaluate_agent(
    agent,
    env,
    n_episodes: int = 100,
    deterministic: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a trained agent over `n_episodes` episodes.

    Parameters
    ----------
    agent : PPOAttackAgent | DQNAttackAgent
        A trained agent with a `run_episode` method.
    env : gym.Env
        Environment to evaluate in.
    n_episodes : int
        Number of evaluation episodes.
    deterministic : bool
        Use deterministic (greedy) policy if True.

    Returns
    -------
    dict with keys:
        rewards, steps, success_rate, catch_rate, avg_detection,
        per_episode  (list of per-episode dicts)
    """
    rewards = []
    steps_list = []
    caught_list = []
    detection_list = []
    goal_reached_list = []
    per_episode = []

    # A reward above this fraction of the total sensitive-host value
    # indicates the agent reached AI infrastructure (not just DMZ/LAN).
    ai_reward_threshold = 100.0

    for ep in range(n_episodes):
        result = agent.run_episode(env, deterministic=deterministic)
        rewards.append(result["total_reward"])
        steps_list.append(result["steps"])
        caught_list.append(int(result.get("caught", False)))
        goal_reached_list.append(int(result["total_reward"] >= ai_reward_threshold))
        if result.get("cumulative_detection") is not None:
            detection_list.append(result["cumulative_detection"])
        per_episode.append(result)

    summary = {
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list)),
        "catch_rate": float(np.mean(caught_list)),
        "success_rate": float(np.mean(goal_reached_list)),
        "mean_cumulative_detection": float(np.mean(detection_list)) if detection_list else None,
        "per_episode": per_episode,
    }
    return summary


def evaluate_both_agents(
    ppo_model_path: str,
    dqn_model_path: str,
    scenario_name: str | None = None,
    stealth: bool = False,
    n_episodes: int = 100,
    detection_threshold: float = 0.8,
    detection_cost_per_step: float = 0.1,
    caught_penalty: float = -100.0,
    alpha: float = 1.0,
    output_dir: str = "results",
) -> dict[str, Any]:
    """
    Load and evaluate both PPO and DQN agents, then save a comparison.

    Parameters
    ----------
    ppo_model_path : str
        Path to the saved PPO model (.zip).
    dqn_model_path : str
        Path to the saved DQN model (.zip).

    Returns
    -------
    dict  {'ppo': eval_summary, 'dqn': eval_summary}
    """
    ppo_env = ActionMaskWrapper(IntActionWrapper(make_env(scenario_name=scenario_name)))
    dqn_env = ActionMaskWrapper(IntActionWrapper(make_env(scenario_name=scenario_name)))
    if stealth:
        ppo_env = make_stealth_env(
            ppo_env,
            detection_threshold=detection_threshold,
            detection_cost_per_step=detection_cost_per_step,
            caught_penalty=caught_penalty,
            alpha=alpha,
        )
        dqn_env = make_stealth_env(
            dqn_env,
            detection_threshold=detection_threshold,
            detection_cost_per_step=detection_cost_per_step,
            caught_penalty=caught_penalty,
            alpha=alpha,
        )

    # Load agents
    ppo_agent = PPOAttackAgent.load(ppo_model_path, ppo_env)
    dqn_agent = DQNAttackAgent.load(dqn_model_path, dqn_env)

    # Evaluate
    print(f"\nEvaluating PPO over {n_episodes} episodes …")
    ppo_results = evaluate_agent(ppo_agent, ppo_env, n_episodes=n_episodes)

    print(f"Evaluating DQN over {n_episodes} episodes …")
    dqn_results = evaluate_agent(dqn_agent, dqn_env, n_episodes=n_episodes)

    # Print comparison table
    _print_comparison(ppo_results, dqn_results, stealth=stealth)

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    combined = {"ppo": ppo_results, "dqn": dqn_results}

    tag = "stealth" if stealth else "baseline"
    eval_file = out / f"eval_{tag}.json"
    with open(eval_file, "w") as f:
        stripped = {
            k: {kk: vv for kk, vv in v.items() if kk != "per_episode"} for k, v in combined.items()
        }
        json.dump(stripped, f, indent=2)

    # Also write to eval_results.json for backward compat
    with open(out / "eval_results.json", "w") as f:
        json.dump(stripped, f, indent=2)

    print(f"\nEvaluation results saved → {eval_file}")

    ppo_env.close()
    dqn_env.close()
    return combined


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def _print_comparison(ppo: dict, dqn: dict, stealth: bool = False) -> None:
    mode = "STEALTH" if stealth else "BASELINE"
    print(f"\n{'='*60}")
    print(f"  Evaluation Comparison ({mode})")
    print(f"{'='*60}")
    header = f"{'Metric':<30} {'PPO':>12} {'DQN':>12}"
    print(header)
    print("-" * 60)

    metrics = [
        ("Mean Reward", "mean_reward", ".2f"),
        ("Std Reward", "std_reward", ".2f"),
        ("Mean Steps", "mean_steps", ".1f"),
        ("Goal Reached (%)", "success_rate", ".1%"),
        ("Catch Rate (%)", "catch_rate", ".1%"),
        ("Mean Detection", "mean_cumulative_detection", ".3f"),
    ]

    for label, key, fmt in metrics:
        ppo_val = ppo.get(key)
        dqn_val = dqn.get(key)
        ppo_str = f"{ppo_val:{fmt}}" if ppo_val is not None else "N/A"
        dqn_str = f"{dqn_val:{fmt}}" if dqn_val is not None else "N/A"
        print(f"{label:<30} {ppo_str:>12} {dqn_str:>12}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained PPO / DQN agents on NASim.")
    p.add_argument("--ppo_model", required=True, help="Path to PPO model (.zip).")
    p.add_argument("--dqn_model", required=True, help="Path to DQN model (.zip).")
    p.add_argument(
        "--scenario",
        default=None,
        help="NASim built-in scenario name. Omit for custom AI-infra topology.",
    )
    p.add_argument(
        "--stealth",
        action="store_true",
        help="Evaluate with stealth wrapper active.",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per agent (default: 100).",
    )
    p.add_argument(
        "--detection_threshold",
        type=float,
        default=0.8,
        help="Detection threshold (default: 0.8).",
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
        "--output_dir",
        default="results",
        help="Directory to save evaluation results (default: results/).",
    )
    return p.parse_args()


def main() -> None:
    """CLI entry point (also registered as ``eval-agent`` console script)."""
    args = _parse_args()
    evaluate_both_agents(
        ppo_model_path=args.ppo_model,
        dqn_model_path=args.dqn_model,
        scenario_name=args.scenario,
        stealth=args.stealth,
        n_episodes=args.episodes,
        detection_threshold=args.detection_threshold,
        detection_cost_per_step=args.detection_cost,
        caught_penalty=args.caught_penalty,
        alpha=args.alpha,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
