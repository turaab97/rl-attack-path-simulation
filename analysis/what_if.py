"""
what_if.py
----------
Counterfactual topology analysis: modify the network configuration
(add/remove firewall rules) and re-evaluate a trained agent to measure
the security impact of network changes.

Author: Syed Ali Turab
Course: MMAI 845 -- Reinforcement Learning

This answers the question security teams care about most: "If we add
this firewall rule, does it actually reduce attacker success?"

Usage:
    from analysis.what_if import evaluate_with_modified_topology

    results = evaluate_with_modified_topology(
        agent=trained_agent,
        modification="block_ssh_to_ai",
        n_episodes=100,
    )
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import Any

import nasim

from environments.network_config import _NETWORK_YAML
from environments.stealth_wrapper import make_stealth_env


# ---------------------------------------------------------------------------
# Topology modifications
# ---------------------------------------------------------------------------
# Each modification is a function that takes the base YAML string and returns
# a modified YAML string. This keeps topology changes declarative and auditable.

def _block_ssh_to_ai(yaml_str: str) -> str:
    """Block SSH from Corporate LAN to AI Infrastructure subnet.

    Removes SSH from the (2, 3) firewall rule, meaning Corporate LAN
    can only reach AI Infrastructure via HTTP. Forces the attacker to
    find an HTTP-only pivot chain.
    """
    return yaml_str.replace(
        "  (2, 3): [http, ssh]",
        "  (2, 3): [http]  # SSH BLOCKED by what-if analysis",
    )


def _block_http_to_ai(yaml_str: str) -> str:
    """Block HTTP from Corporate LAN to AI Infrastructure subnet.

    Removes HTTP from the (2, 3) firewall rule, meaning Corporate LAN
    can only reach AI Infrastructure via SSH. Forces the attacker to
    find an SSH-only pivot chain.
    """
    return yaml_str.replace(
        "  (2, 3): [http, ssh]",
        "  (2, 3): [ssh]  # HTTP BLOCKED by what-if analysis",
    )


def _isolate_data_lake(yaml_str: str) -> str:
    """Fully isolate the Data Lake from AI Infrastructure.

    Removes the (3, 4) firewall rule entirely, cutting off all subnet-level
    access from AI Infrastructure to the Data Lake. The attacker cannot
    reach the highest-value target.
    """
    return yaml_str.replace(
        "  (3, 4): [ssh]",
        "  (3, 4): []  # ALL ACCESS BLOCKED by what-if analysis",
    )


def _add_segmentation(yaml_str: str) -> str:
    """Add network segmentation: block direct Corporate LAN to AI paths.

    Removes ALL services from the (2, 3) firewall rule, cutting off the
    only direct path from Corporate LAN to AI Infrastructure.
    """
    return yaml_str.replace(
        "  (2, 3): [http, ssh]",
        "  (2, 3): []  # ALL ACCESS BLOCKED by segmentation",
    )


MODIFICATIONS = {
    "block_ssh_to_ai": {
        "fn": _block_ssh_to_ai,
        "description": (
            "Block SSH from Internal Services (2,3) to LLM API Server (3,0). "
            "Simulates adding a firewall rule that restricts SSH lateral "
            "movement into the AI infrastructure subnet."
        ),
    },
    "block_http_to_ai": {
        "fn": _block_http_to_ai,
        "description": (
            "Block HTTP from Dev Server (2,1) to LLM API Server (3,0). "
            "Simulates restricting web-based access to AI infrastructure."
        ),
    },
    "isolate_data_lake": {
        "fn": _isolate_data_lake,
        "description": (
            "Remove SSH access from Model Repo (3,2) to Training Data Lake (4,0). "
            "Simulates tighter segmentation around the highest-value asset."
        ),
    },
    "full_segmentation": {
        "fn": _add_segmentation,
        "description": (
            "Block ALL direct paths from Corporate LAN to AI Infrastructure. "
            "Simulates a zero-trust segmentation policy between corporate "
            "and AI network zones."
        ),
    },
}


# ---------------------------------------------------------------------------
# Evaluation with modified topology
# ---------------------------------------------------------------------------


def make_modified_env(modification: str, stealth: bool = False, **stealth_kwargs):
    """Create a NASim environment with a modified topology.

    Parameters
    ----------
    modification : str
        One of the keys in MODIFICATIONS.
    stealth : bool
        Whether to wrap with stealth-aware reward.

    Returns
    -------
    gym.Env
        The modified NASim environment.
    """
    if modification not in MODIFICATIONS:
        available = ", ".join(MODIFICATIONS.keys())
        raise ValueError(
            f"Unknown modification: {modification!r}. Choose from: {available}"
        )

    mod_fn = MODIFICATIONS[modification]["fn"]
    modified_yaml = mod_fn(_NETWORK_YAML)

    fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(modified_yaml)
        env = nasim.load(tmp_path, name=f"ai-infra-{modification}", fully_obs=True)
    finally:
        os.unlink(tmp_path)

    if stealth:
        env = make_stealth_env(env, **stealth_kwargs)

    return env


def evaluate_with_modified_topology(
    agent,
    modification: str,
    n_episodes: int = 100,
    deterministic: bool = True,
    stealth: bool = False,
    **stealth_kwargs,
) -> dict[str, Any]:
    """Evaluate a trained agent on a modified network topology.

    The agent is NOT retrained -- this measures how a network change
    affects an attacker who has already learned the original topology.
    This is the realistic scenario: the attacker has reconnaissance
    knowledge of the old network, and the defender deploys a change.

    Parameters
    ----------
    agent : PPOAttackAgent | DQNAttackAgent
        A trained agent (trained on the original topology).
    modification : str
        Which topology modification to apply.
    n_episodes : int
        Number of evaluation episodes.
    deterministic : bool
        Use greedy policy.
    stealth : bool
        Whether to apply stealth wrapper to the modified env.

    Returns
    -------
    dict with keys:
        modification, description, mean_reward, success_rate,
        catch_rate, mean_steps, n_episodes
    """
    import numpy as np

    env = make_modified_env(modification, stealth=stealth, **stealth_kwargs)

    rewards = []
    steps_list = []
    caught_list = []
    ai_threshold = 100.0

    for _ in range(n_episodes):
        result = agent.run_episode(env, deterministic=deterministic)
        rewards.append(result["total_reward"])
        steps_list.append(result["steps"])
        caught_list.append(int(result.get("caught", False)))

    env.close()

    return {
        "modification": modification,
        "description": MODIFICATIONS[modification]["description"],
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean([r >= ai_threshold for r in rewards])),
        "catch_rate": float(np.mean(caught_list)),
        "mean_steps": float(np.mean(steps_list)),
    }


def run_all_what_if(
    agent,
    n_episodes: int = 100,
    deterministic: bool = True,
    stealth: bool = False,
    **stealth_kwargs,
) -> list[dict[str, Any]]:
    """Run all available topology modifications and return comparison data.

    Parameters
    ----------
    agent : PPOAttackAgent | DQNAttackAgent
        Trained agent.

    Returns
    -------
    list[dict]
        One result dict per modification, suitable for building a
        comparison table.
    """
    results = []
    for mod_name in MODIFICATIONS:
        print(f"  Evaluating what-if: {mod_name} ...")
        result = evaluate_with_modified_topology(
            agent=agent,
            modification=mod_name,
            n_episodes=n_episodes,
            deterministic=deterministic,
            stealth=stealth,
            **stealth_kwargs,
        )
        results.append(result)
    return results
