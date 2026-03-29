"""
network_config.py
-----------------
Defines the NASim scenario representing a corporate network that hosts
AI infrastructure (LLM servers, vector databases, model repositories).

Author: Syed Ali Turab
Course: MMAI 845 – Reinforcement Learning

Network topology
================
Subnet 0 (Internet / attacker entry point)
  └─ Subnet 1  DMZ  – public-facing web servers, email gateway
       └─ Subnet 2  Corporate LAN – workstations, AD, internal services
            └─ Subnet 3  AI Infrastructure – LLM servers, vector DB, model repo
                 └─ Subnet 4  Data Lake – training data stores (highest value)

Every host in Subnet 3-4 is tagged as an AI-infrastructure asset.
The attacker starts in Subnet 0 and must pivot through the subnets.

This file exposes:
  - build_network_scenario()  →  nasim.Scenario object
  - AI_INFRA_HOSTS            →  list of (subnet, host) tuples
  - make_env()                →  nasim.NASimEnv ready to use
"""

from __future__ import annotations

import os
import tempfile

import nasim

# ---------------------------------------------------------------------------
# Host catalogue
# ---------------------------------------------------------------------------
# Each host entry follows the NASim YAML schema:
#   os, services, processes, value, firewall, access
# ---------------------------------------------------------------------------

# High-value target hosts – AI infrastructure
AI_INFRA_HOSTS = [
    (3, 0),  # LLM API Server
    (3, 1),  # Vector Database (Pinecone / Weaviate clone)
    (3, 2),  # Model Repository (MLflow / DVC server)
    (4, 0),  # Training Data Lake (highest value)
]

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
