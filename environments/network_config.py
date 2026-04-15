"""
network_config.py
-----------------
Defines the NASim scenario representing a corporate network that hosts
AI infrastructure (LLM servers, vector databases, model repositories).

Author: Syed Ali Turab
Course: MMAI 845 -- Reinforcement Learning

Network topology
================
This file defines the RL *environment* -- the world the agent interacts
with.  In MDP terms:

  - **State**: NASim generates a flat observation vector encoding each
    host's discovery status, access level, services, and OS type.
  - **Actions**: NASim enumerates all (target_host, action_type) pairs.
  - **Transitions**: exploits succeed with fixed probability (0.7-0.9);
    failures still cost -1.
  - **Rewards**: positive on first compromise of a sensitive host (the
    values below), -1 per step otherwise.

The topology is a 5-subnet linear chain modelling a real enterprise:

  Subnet 0 (Internet / attacker entry point)
    |
    v
  Subnet 1  DMZ  -- public-facing web servers, email gateway
    |
    v
  Subnet 2  Corporate LAN -- workstations, AD, internal services
    |
    v
  Subnet 3  AI Infrastructure -- LLM servers, vector DB, model repo
    |
    v
  Subnet 4  Data Lake -- training data stores (highest value)

The attacker starts in Subnet 0 and must discover hosts, exploit
services, and escalate privileges across each firewall boundary to
reach the AI infrastructure.

This file exposes:
  - build_network_scenario()  ->  nasim.Scenario object
  - AI_INFRA_HOSTS            ->  list of (subnet, host) tuples
  - make_env()                ->  nasim.NASimEnv ready to use
"""

from __future__ import annotations

import os
import tempfile

import nasim

# ---------------------------------------------------------------------------
# High-value targets -- these are the AI-infrastructure hosts the attacker
# must reach.  The tuple format is (subnet_index, host_index).  Subnet 3
# houses the AI compute/serving layer; Subnet 4 is the data lake.
# These coordinates are used by the evaluation harness to check whether the
# agent's attack path actually reached AI infrastructure.
# ---------------------------------------------------------------------------
AI_INFRA_HOSTS = [
    (3, 0),  # LLM API Server       -- serves model inference requests
    (3, 1),  # Vector Database       -- stores embeddings (e.g. Pinecone)
    (3, 2),  # Model Repository      -- stores trained models (e.g. MLflow)
    (4, 0),  # Training Data Lake    -- raw training data (highest value)
]

# ---------------------------------------------------------------------------
# NASim YAML scenario definition
# ---------------------------------------------------------------------------
# NASim reads a YAML file to build the MDP.  Key sections:
#
#   subnets          -- number of hosts per subnet (NASim auto-adds Subnet 0
#                       as the internet/attacker entry; these counts are for
#                       Subnets 1-4 only).
#   topology         -- 5x5 adjacency matrix; 1 = subnets are connected.
#   sensitive_hosts  -- (subnet, host): reward granted on first compromise.
#   exploits         -- available attack actions with service/OS requirements,
#                       success probability, cost, and access level gained.
#   privilege_escalation -- post-exploit actions to escalate from user to root.
#   host_configurations  -- per-host OS, services, and processes.
#   firewall         -- (src_subnet, dst_subnet): services allowed through.
#   step_limit       -- maximum episode length (horizon of the MDP).
# ---------------------------------------------------------------------------
_NETWORK_YAML = """\
subnets: [3, 4, 3, 1]

topology: [
  [1, 1, 0, 0, 0],
  [1, 1, 1, 0, 0],
  [0, 1, 1, 1, 0],
  [0, 0, 1, 1, 1],
  [0, 0, 0, 1, 1]
]

sensitive_hosts:
  (3, 0): 200
  (3, 1): 200
  (3, 2): 150
  (4, 0): 300

os:
  - linux
  - windows

services:
  - ssh
  - http
  - smb
  - rdp

processes:
  - apache
  - sshd
  - smbd

exploits:
  e_ssh:
    service: ssh
    os: linux
    prob: 0.8
    cost: 1
    access: user
  e_http:
    service: http
    os: None
    prob: 0.7
    cost: 1
    access: user
  e_smb:
    service: smb
    os: windows
    prob: 0.9
    cost: 1
    access: user
  e_rdp:
    service: rdp
    os: windows
    prob: 0.7
    cost: 1
    access: user

privilege_escalation:
  pe_linux:
    process: apache
    os: linux
    prob: 0.8
    cost: 1
    access: root
  pe_windows:
    process: smbd
    os: windows
    prob: 0.7
    cost: 1
    access: root

service_scan_cost: 1
os_scan_cost: 1
subnet_scan_cost: 1
process_scan_cost: 1

host_configurations:
  (1, 0):
    os: linux
    services: [http, ssh]
    processes: [apache, sshd]
  (1, 1):
    os: windows
    services: [smb, rdp]
    processes: [smbd]
  (1, 2):
    os: linux
    services: [ssh, http]
    processes: [sshd, apache]
  (2, 0):
    os: windows
    services: [smb, rdp]
    processes: [smbd]
  (2, 1):
    os: linux
    services: [ssh, http]
    processes: [sshd, apache]
  (2, 2):
    os: windows
    services: [smb, rdp]
    processes: [smbd]
  (2, 3):
    os: linux
    services: [ssh]
    processes: [sshd]
  (3, 0):
    os: linux
    services: [http, ssh]
    processes: [apache, sshd]
  (3, 1):
    os: linux
    services: [http]
    processes: [apache]
  (3, 2):
    os: linux
    services: [ssh]
    processes: [sshd]
  (4, 0):
    os: linux
    services: [ssh]
    processes: [sshd]

firewall:
  (0, 1): [http, ssh, smb, rdp]
  (1, 0): []
  (1, 2): [ssh, smb, rdp, http]
  (2, 1): [ssh, http]
  (2, 3): [http, ssh]
  (3, 2): [ssh]
  (3, 4): [ssh]
  (4, 3): [ssh]

step_limit: 500
"""


def build_network_scenario() -> nasim.Scenario:
    """
    Build and return a NASim Scenario for the corporate AI-infrastructure
    network.

    Returns
    -------
    nasim.Scenario
        A fully-configured scenario ready to be wrapped with nasim.load().
        Stealth behaviour is handled separately by StealthAwareWrapper.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(_NETWORK_YAML)
        scenario = nasim.load_scenario(tmp_path)
    finally:
        os.unlink(tmp_path)
    return scenario


def make_env(
    scenario_name: str | None = None,
    stealth: bool = False,
    fully_obs: bool = True,
) -> nasim.NASimEnv:
    """
    Convenience factory: returns a ready-to-use NASim environment.

    Parameters
    ----------
    scenario_name : str or None
        If provided, loads a built-in NASim benchmark scenario (e.g.
        'small-linear', 'medium', etc.).  If None, uses the custom
        AI-infrastructure topology defined in this file.
    stealth : bool
        Whether to tag the environment for stealth-mode training.
    fully_obs : bool
        If True (default), the agent observes the full network state from
        step 0.  If False, the agent starts with no knowledge and must scan
        to discover hosts (much harder to learn).

    Returns
    -------
    nasim.NASimEnv
    """
    if scenario_name is not None:
        env = nasim.make_benchmark(scenario_name, fully_obs=fully_obs)
    else:
        fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(_NETWORK_YAML)
            env = nasim.load(tmp_path, name="ai-infra", fully_obs=fully_obs)
        finally:
            os.unlink(tmp_path)
    return env
