"""
network_config.py
-----------------
Defines the NASim scenario representing a corporate network that hosts
AI infrastructure (LLM servers, vector databases, model repositories).

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
  - build_network_scenario(stealth=False)  →  nasim.Scenario object
  - AI_INFRA_HOSTS                         →  list of (subnet, host) tuples
"""

from __future__ import annotations
import yaml
import nasim

# ---------------------------------------------------------------------------
# Host catalogue
# ---------------------------------------------------------------------------
# Each host entry follows the NASim YAML schema:
#   os, services, processes, value, firewall, access
# ---------------------------------------------------------------------------

# High-value target hosts – AI infrastructure
AI_INFRA_HOSTS = [
    (3, 0),   # LLM API Server
    (3, 1),   # Vector Database (Pinecone / Weaviate clone)
    (3, 2),   # Model Repository (MLflow / DVC server)
    (4, 0),   # Training Data Lake (highest value)
]

_NETWORK_YAML = """\
subnets: [1, 3, 4, 3, 1]

topology: [
  [1, 1, 0, 0, 0],
  [1, 1, 1, 0, 0],
  [0, 1, 1, 1, 0],
  [0, 0, 1, 1, 1],
  [0, 0, 0, 1, 1]
]

sensitive_hosts:
  (3, 0): 200   # LLM API Server
  (3, 1): 200   # Vector DB
  (3, 2): 150   # Model Repo
  (4, 0): 300   # Training Data Lake

os:
  linux: 0.5
  windows: 0.5

services:
  ssh: 0.5
  http: 0.4
  smb: 0.3
  rdp: 0.3

processes:
  apache: 0.4
  sshd: 0.5
  smbd: 0.3

exploits:
  e_ssh:
    service: ssh
    os: linux
    prob: 0.8
    cost: 1
    access: user
  e_http:
    service: http
    os: linux
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

host_configurations:
  # --- DMZ (subnet 1) ---
  (1, 0):
    os: linux
    services: [http, ssh]
    processes: [apache, sshd]
    value: 0
    firewall: {}
    access: none
  (1, 1):
    os: windows
    services: [smb, rdp]
    processes: [smbd]
    value: 0
    firewall: {}
    access: none
  (1, 2):
    os: linux
    services: [ssh, http]
    processes: [sshd, apache]
    value: 0
    firewall: {}
    access: none

  # --- Corporate LAN (subnet 2) ---
  (2, 0):
    os: windows
    services: [smb, rdp]
    processes: [smbd]
    value: 10
    firewall:
      (1, 0): [smb]
      (1, 1): [smb, rdp]
    access: none
  (2, 1):
    os: linux
    services: [ssh, http]
    processes: [sshd, apache]
    value: 10
    firewall:
      (1, 2): [ssh]
    access: none
  (2, 2):
    os: windows
    services: [smb, rdp]
    processes: [smbd]
    value: 20
    firewall:
      (2, 0): [smb]
    access: none
  (2, 3):
    os: linux
    services: [ssh]
    processes: [sshd]
    value: 10
    firewall:
      (2, 1): [ssh]
    access: none

  # --- AI Infrastructure (subnet 3) ---
  (3, 0):
    os: linux
    services: [http, ssh]
    processes: [apache, sshd]
    value: 200
    firewall:
      (2, 1): [http]
      (2, 3): [ssh]
    access: none
  (3, 1):
    os: linux
    services: [http]
    processes: [apache]
    value: 200
    firewall:
      (3, 0): [http]
    access: none
  (3, 2):
    os: linux
    services: [ssh]
    processes: [sshd]
    value: 150
    firewall:
      (3, 0): [ssh]
    access: none

  # --- Data Lake (subnet 4) ---
  (4, 0):
    os: linux
    services: [ssh]
    processes: [sshd]
    value: 300
    firewall:
      (3, 0): [ssh]
      (3, 2): [ssh]
    access: none
"""


def build_network_scenario(
    stealth: bool = False,
    detection_threshold: float = 0.8,
) -> nasim.Scenario:
    """
    Build and return a NASim Scenario for the corporate AI-infrastructure
    network.

    Parameters
    ----------
    stealth : bool
        If True, a non-zero detection cost is embedded in each exploit so the
        stealth wrapper can accumulate risk correctly.  (The wrapper itself
        adjusts the reward; this flag mainly serves as documentation.)
    detection_threshold : float
        Passed through for reference; enforcement lives in StealthAwareWrapper.

    Returns
    -------
    nasim.Scenario
        A fully-configured scenario ready to be wrapped with nasim.load().
    """
    scenario_dict = yaml.safe_load(_NETWORK_YAML)
    scenario = nasim.load_scenario(scenario_dict)
    return scenario


def make_env(scenario_name: str | None = None, stealth: bool = False) -> nasim.NASimEnv:
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

    Returns
    -------
    nasim.NASimEnv
    """
    if scenario_name is not None:
        env = nasim.make(scenario_name)
    else:
        scenario = build_network_scenario(stealth=stealth)
        env = nasim.load(scenario)
    return env
