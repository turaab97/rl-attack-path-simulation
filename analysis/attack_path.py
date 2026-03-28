"""
attack_path.py
--------------
Utilities for interpreting and analysing RL agent attack paths.

Author: Syed Ali Turab
Course: MMAI 845 – Reinforcement Learning

Maps raw NASim action indices back to human-readable (subnet, host, action_type)
tuples so security teams can understand the actual pivot chain the agent used.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# NASim action-space mapping
# ---------------------------------------------------------------------------

# NASim action ordering (flat discrete):
#   For each (subnet, host): [noop, scan, exploit_1, ..., exploit_n, priv_esc_1, ..., priv_esc_m]
# The exact mapping depends on the scenario, so we build it dynamically.

SUBNET_NAMES = {
    0: "Internet (Attacker)",
    1: "DMZ",
    2: "Corporate LAN",
    3: "AI Infrastructure",
    4: "Data Lake",
}

HOST_NAMES = {
    (1, 0): "Web Server",
    (1, 1): "Email Gateway",
    (1, 2): "DNS Server",
    (2, 0): "Workstation-A",
    (2, 1): "Dev Server",
    (2, 2): "Active Directory",
    (2, 3): "Internal Services",
    (3, 0): "LLM API Server",
    (3, 1): "Vector Database",
    (3, 2): "Model Repository",
    (4, 0): "Training Data Lake",
}


def build_action_map(env) -> list[dict[str, Any]]:
    """Build a mapping from flat action index to structured action info.

    Parameters
    ----------
    env : nasim.NASimEnv
        The NASim environment (unwrapped).

    Returns
    -------
    list[dict]
        One dict per action index with keys:
        action_idx, subnet, host, action_type, action_name, host_name
    """
    action_map = []

    # NASim stores action descriptions accessible via the scenario
    try:
        n_actions = env.action_space.n
    except AttributeError:
        unwrapped = env.unwrapped
        n_actions = unwrapped.action_space.n

    # Try to get action descriptions from NASim's internal API
    unwrapped = getattr(env, "unwrapped", env)
    scenario = getattr(unwrapped, "scenario", None)

    if scenario is not None and hasattr(scenario, "actions"):
        actions = scenario.actions
        for idx in range(min(n_actions, len(actions))):
            act = actions[idx]
            # NASim actions have: name, target (subnet, host), cost
            subnet = act.get("subnet", -1) if isinstance(act, dict) else getattr(act, "subnet", -1)
            host = act.get("host", -1) if isinstance(act, dict) else getattr(act, "host", -1)
            name = (
                act.get("name", "unknown")
                if isinstance(act, dict)
                else getattr(act, "name", "unknown")
            )

            action_type = _classify_action(name)
            host_key = (subnet, host)
            action_map.append(
                {
                    "action_idx": idx,
                    "subnet": subnet,
                    "host": host,
                    "action_type": action_type,
                    "action_name": name,
                    "host_name": HOST_NAMES.get(host_key, f"host-{subnet}-{host}"),
                    "subnet_name": SUBNET_NAMES.get(subnet, f"subnet-{subnet}"),
                }
            )
    else:
        # Fallback: generate generic labels
        for idx in range(n_actions):
            action_map.append(
                {
                    "action_idx": idx,
                    "subnet": -1,
                    "host": -1,
                    "action_type": "unknown",
                    "action_name": f"action_{idx}",
                    "host_name": "unknown",
                    "subnet_name": "unknown",
                }
            )

    return action_map


def _classify_action(name: str) -> str:
    """Classify an action name into scan / exploit / privilege_escalation / noop."""
    n = name.lower()
    if "noop" in n or "no-op" in n or n == "":
        return "noop"
    if "scan" in n or "subnet_scan" in n or "os_scan" in n or "service_scan" in n:
        return "scan"
    if "pe_" in n or "priv" in n:
        return "privilege_escalation"
    if "e_" in n or "exploit" in n:
        return "exploit"
    return "other"


# ---------------------------------------------------------------------------
# Path analysis
# ---------------------------------------------------------------------------


def interpret_path(
    path: list[int],
    action_map: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert a raw action-index path to a human-readable sequence.

    Parameters
    ----------
    path : list[int]
        Sequence of action indices from agent.run_episode().
    action_map : list[dict]
        Output of build_action_map().

    Returns
    -------
    list[dict]
        One dict per step with step number and action details.
    """
    interpreted = []
    for step_num, action_idx in enumerate(path):
        if action_idx < len(action_map):
            entry = dict(action_map[action_idx])
        else:
            entry = {
                "action_idx": action_idx,
                "action_type": "unknown",
                "action_name": f"action_{action_idx}",
                "host_name": "unknown",
                "subnet_name": "unknown",
            }
        entry["step"] = step_num
        interpreted.append(entry)
    return interpreted


def summarise_path(interpreted_path: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce aggregate statistics from an interpreted attack path.

    Returns
    -------
    dict with keys:
        total_steps, action_type_counts, hosts_targeted, subnets_visited,
        pivot_chain (ordered unique subnets visited)
    """
    action_types = Counter(step["action_type"] for step in interpreted_path)
    hosts_targeted = Counter(step["host_name"] for step in interpreted_path)
    subnets = []
    for step in interpreted_path:
        sn = step.get("subnet_name", "unknown")
        if sn != "unknown" and (not subnets or subnets[-1] != sn):
            subnets.append(sn)

    return {
        "total_steps": len(interpreted_path),
        "action_type_counts": dict(action_types),
        "hosts_targeted": dict(hosts_targeted),
        "subnets_visited": list(set(s.get("subnet_name", "?") for s in interpreted_path)),
        "pivot_chain": subnets,
    }


def find_common_pivots(
    paths: list[list[int]],
    action_map: list[dict[str, Any]],
    top_n: int = 5,
) -> list[tuple[str, int]]:
    """Identify which hosts appear most often across multiple attack paths.

    This directly answers the security question: "Which hosts serve as
    stepping stones to AI infrastructure?"

    Parameters
    ----------
    paths : list[list[int]]
        Multiple raw action paths from evaluation episodes.
    action_map : list[dict]
        Output of build_action_map().
    top_n : int
        Number of top pivot hosts to return.

    Returns
    -------
    list[tuple[str, int]]
        (host_name, count) sorted by frequency descending.
    """
    host_counts: Counter = Counter()
    for path in paths:
        interpreted = interpret_path(path, action_map)
        active_hosts = set()
        for step in interpreted:
            if step["action_type"] in ("exploit", "privilege_escalation"):
                active_hosts.add(step["host_name"])
        for h in active_hosts:
            host_counts[h] += 1

    return host_counts.most_common(top_n)
