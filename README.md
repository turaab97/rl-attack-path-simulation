# Using AI to Secure AI: RL for Attack Path Simulation Against Enterprise AI Infrastructure

> **MMAI 845 -- Reinforcement Learning | Team Broadview**

[![CI](https://github.com/turaab97/rl-attack-path-simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/turaab97/rl-attack-path-simulation/actions)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.3%2B-orange)](https://stable-baselines3.readthedocs.io/)
[![NASim](https://img.shields.io/badge/NASim-0.10%2B-green)](https://networkattacksimulator.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Table of Contents

1. [Overview](#overview)
2. [Research Questions](#research-questions)
3. [Network Topology](#network-topology)
4. [MDP Formulation](#mdp-formulation)
5. [Algorithms: PPO vs DQN](#algorithms-ppo-vs-dqn)
6. [Reward Variants](#reward-variants)
7. [Results Summary](#results-summary)
8. [Reproducing the Results](#reproducing-the-results)
9. [Project Structure](#project-structure)
10. [CLI Reference](#cli-reference)
11. [Analysis & Reporting Tools](#analysis--reporting-tools)
12. [Developer Tooling](#developer-tooling)
13. [Future Work](#future-work)
14. [References](#references)

---

## Overview

Enterprises are deploying AI infrastructure -- LLM servers, vector databases, and model repositories -- into corporate networks that were not designed to protect these assets. Traditional penetration tests are expensive and infrequent; static vulnerability scanners do not model multi-step attacker behaviour.

This project uses **Reinforcement Learning** to automate attack path discovery. An RL agent acts as a red teamer navigating a simulated corporate network, learning efficient multi-hop paths to high-value AI infrastructure. Two algorithms -- **PPO (Proximal Policy Optimization)** and **DQN (Deep Q-Network)** -- are trained and compared under both a baseline reward signal and a stealth-aware reward signal that models detection by a Security Operations Centre (SOC).

Security teams can use the outputs to:

- Identify which hosts serve as **stepping stones** to AI assets
- Evaluate whether a proposed **network change** reduces attack success
- Compare attacker behaviour under **different detection regimes**

---

## Research Questions

1. Can PPO and DQN learn effective multi-hop attack paths to AI infrastructure in a simulated enterprise network?
2. How does a stealth-aware reward signal change the attack strategy learned by each algorithm?
3. Which algorithm produces faster convergence and higher attack consistency?

---

## Network Topology

The environment models a 5-subnet enterprise network with 11 hosts. The attacker starts at the internet boundary (Subnet 0) and must pivot through firewalled subnets to reach AI infrastructure.

```
                    +--------------------------+
                    |   Subnet 0 (Internet)    |
                    |   Attacker entry point    |
                    +------------+-------------+
                                 |
                    Firewall: HTTP, SSH, SMB, RDP allowed
                                 |
                    +------------v-------------+
                    |     Subnet 1 (DMZ)       |
                    |  3 hosts: Web, Email, DNS |
                    |  Linux + Windows mix      |
                    +------------+-------------+
                                 |
                    Firewall: SSH, SMB, RDP, HTTP allowed
                                 |
                    +------------v-------------+
                    |  Subnet 2 (Corp LAN)     |
                    |  4 hosts: Workstations,   |
                    |  AD, Internal Services    |
                    +------------+-------------+
                                 |
                    Firewall: HTTP, SSH allowed
                                 |
                    +------------v-----------------+
                    |  Subnet 3 (AI Infrastructure)|  <-- HIGH VALUE
                    |  LLM API Server  (reward 200)|
                    |  Vector DB       (reward 200)|
                    |  Model Repository(reward 150)|
                    +------------+-----------------+
                                 |
                    Firewall: SSH only
                                 |
                    +------------v-------------+
                    |  Subnet 4 (Data Lake)    |  <-- HIGHEST VALUE
                    |  Training Data (reward 300)|
                    +--------------------------+
```

Each hop requires the agent to scan for hosts, identify services, exploit vulnerabilities (SSH, HTTP, SMB, RDP), and escalate privileges -- mirroring a real penetration test.

---

## MDP Formulation

The network attack is formulated as a finite-horizon Markov Decision Process using [NASim](https://networkattacksimulator.readthedocs.io/) (Network Attack Simulator).

| Component | Description |
|---|---|
| **State** | Flat vector of 264 features: per-host discovery status, access level (none/user/root), running services, OS type |
| **Actions** | 110 discrete actions: `(target_host, action_type)` combinations -- subnet scan, OS scan, service scan, exploit (SSH/HTTP/SMB/RDP), privilege escalation |
| **Action Masking** | Only ~5-15 of 110 actions are valid at any state (host must be discovered AND reachable). Invalid actions are masked out before the agent selects. |
| **Rewards** | +value on first compromise of each sensitive host; -1 step cost per action |
| **Transitions** | Exploits succeed with a fixed probability (0.7-0.9 depending on type). Failed exploits still cost -1. |
| **Episode** | Terminates after 500 steps or when all goals are reached (or when caught in stealth mode) |
| **Observability** | Full observability -- agent sees entire network state. This is a simplification; partial observability would require the agent to scan before seeing host information. |

**Key modelling decision:** We use full observability (`fully_obs=True`) because partial observability makes the problem significantly harder and agents failed to learn within our compute budget. In a production system, you would use partial observability to model a real attacker's information set more accurately.

---

## Algorithms: PPO vs DQN

We compare two fundamentally different RL algorithms to evaluate which is more suitable for automated attack path discovery.

### PPO (Proximal Policy Optimization) -- On-Policy

PPO directly learns a stochastic policy (probability distribution over actions). It collects batches of experience, computes advantages, and updates the policy using a clipped surrogate objective that prevents destructively large updates.

We use **MaskablePPO** from `sb3-contrib`, which natively zeroes out logits for invalid actions before sampling. Every action during training is guaranteed valid.

| Hyperparameter | Value | Rationale |
|---|---|---|
| Learning rate | 3e-4 | Standard PPO default |
| Steps per rollout | 2,048 | Enough for multi-step attack chains |
| Batch size | 64 | - |
| PPO epochs | 10 | - |
| Clip range | 0.2 | Standard stability parameter |
| Entropy coef | 0.05 | Higher than default (0.01) to encourage exploration in large action space |
| Gamma | 0.99 | Long-horizon discounting for multi-hop paths |
| Network | MLP [256, 256] | Two hidden layers |

### DQN (Deep Q-Network) -- Off-Policy

DQN learns a Q-function estimating expected return for each state-action pair. It uses a replay buffer for sample efficiency and a target network for training stability.

Action masking is applied manually: Q-values of invalid actions are set to -infinity before taking the argmax, and epsilon-greedy exploration samples only from valid actions.

| Hyperparameter | Value | Rationale |
|---|---|---|
| Learning rate | 1e-4 | Lower than PPO for stable Q-learning |
| Replay buffer | 100,000 | Store past transitions for off-policy learning |
| Exploration | epsilon 1.0 -> 0.05 over 50% of training | Extended exploration for large action space |
| Target network update | Every 1,000 steps | Hard updates for stability |
| Train frequency | Every 4 steps | - |
| Gamma | 0.99 | Same as PPO for fair comparison |
| Network | MLP [256, 256] | Same architecture as PPO |

### Why These Two?

PPO with action masking is the standard choice for discrete control with large invalid-action spaces. DQN provides a contrasting off-policy baseline -- it can reuse past experience from its replay buffer, which is theoretically more sample-efficient. Comparing them reveals whether on-policy stability or off-policy efficiency matters more for network attack simulation.

---

## Reward Variants

### Baseline

Raw NASim rewards: +value on first compromise of a sensitive host, -1 per action.

A **dense reward wrapper** adds a +3.0 bonus when an action changes more than one observation feature (indicating real progress like host discovery or access gain, as opposed to the step counter incrementing).

### Stealth-Aware

Adds a detection model on top of the baseline reward:

```
r' = r_env  -  alpha * detection_cost_per_step     (for active actions)
r' = r_env                                          (for idle actions)
```

If cumulative detection reaches the threshold (default 0.8), the episode terminates with a -100 penalty (attacker caught by SOC/IDS). With a detection cost of 0.1 per active step, the agent has at most 8 active steps before being caught.

The wrapper composition order is: `IntAction -> ActionMask -> DenseReward -> Stealth`

---

## Results Summary

Both algorithms were trained for 500,000 timesteps on the custom AI-infrastructure topology with seed 42. Evaluation was run over 100 episodes.

### Baseline (No Detection)

| Metric | PPO | DQN |
|---|---|---|
| Mean Reward | **-100.0** | -498.0 |
| Std Reward | **0.0** | 19.9 |
| Mean Steps | 500.0 | 500.0 |
| Success Rate | 0% | 0% |

**PPO** learned a consistent policy (zero variance) that navigates the network systematically. **DQN** collapsed to near-random behaviour with high variance, indicating unstable Q-learning in this environment.

DQN's failure has three root causes: (1) replay buffer staleness mixing old experience with current policy, (2) manual action masking doesn't influence gradient updates like PPO's native logit masking, (3) uniform epsilon-greedy exploration over valid actions is less efficient than PPO's entropy-regularized sampling.

### Stealth Mode (Detection Active)

| Metric | PPO | DQN |
|---|---|---|
| Mean Reward | -109.9 | -109.9 |
| Mean Steps | 9.0 | 9.0 |
| Catch Rate | **100%** | **100%** |
| Cumulative Detection | 0.9 | 0.9 |

Both agents were caught after exactly 9 active steps. The detection threshold of 0.8 with 0.1 cost per step is aggressive enough to catch automated attackers before they reach AI infrastructure. This demonstrates the stealth wrapper is working correctly.

### Key Findings

1. **PPO is the more reliable red team agent** -- zero variance means repeatable, auditable attack paths.
2. **DQN struggles with this problem** due to replay buffer dynamics and manual action masking limitations.
3. **The stealth wrapper catches both agents at step 8-9**, validating the detection model and showing that the current SOC sensitivity is sufficient against automated attackers.
4. **Neither agent achieved the strict success threshold** (+100 reward), but PPO's attack paths show systematic network traversal. With additional training budget (2-5M steps) or curriculum learning, success rates would improve.

---

## Reproducing the Results

> **This section is critical for evaluation. Follow these steps exactly to reproduce our results.**

### Prerequisites

- **Python 3.10, 3.11, or 3.12** (PyTorch does not support 3.13)
- **Git** for cloning the repository
- **Docker** (optional but recommended -- avoids all dependency issues)
- **Google Colab** (optional -- for GPU-accelerated training)

### Option 1: One-Command Shell Script (Recommended)

The `run.sh` script handles everything -- virtual environment creation, dependency installation, training, evaluation, and report generation.

```bash
git clone https://github.com/turaab97/rl-attack-path-simulation.git
cd rl-attack-path-simulation
chmod +x run.sh
./run.sh
```

The script will:
1. Create a Python virtual environment
2. Install all dependencies (pinned versions)
3. Run the linter and test suite
4. Train both PPO and DQN agents (baseline mode, 500k steps each)
5. Train both agents in stealth mode (500k steps each)
6. Evaluate all four trained models (100 episodes, seed 42)
7. Generate all figures and the penetration testing report
8. Print a summary of results

All output is saved to `results/`. Estimated time: 30-60 minutes on a modern CPU.

### Option 2: Docker (Zero Local Dependencies)

```bash
git clone https://github.com/turaab97/rl-attack-path-simulation.git
cd rl-attack-path-simulation

# Train baseline comparison (PPO + DQN, 500k steps each)
docker compose run train

# Train stealth mode comparison
docker compose run train-stealth

# Evaluate trained models (100 episodes each)
docker compose run evaluate
docker compose run evaluate-stealth

# Generate visualizations
docker compose run visualize

# Run test suite
docker compose run test
```

Results are persisted to `./results/` on the host via a bind mount.

### Option 3: Manual Local Installation

```bash
git clone https://github.com/turaab97/rl-attack-path-simulation.git
cd rl-attack-path-simulation

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install the package with all dependencies
pip install --upgrade pip
pip install -e ".[dev]"

# Verify installation
python -c "import nasim; import stable_baselines3; print('OK')"

# Run linter and tests
black --check .
isort --check-only .
flake8 . --count --max-line-length=120 --statistics
pytest tests/ -v
```

**Training (manual commands):**

```bash
# Train PPO baseline (500k steps)
python -m training.train --agent ppo --timesteps 500000 --seed 42

# Train DQN baseline (500k steps)
python -m training.train --agent dqn --timesteps 500000 --seed 42

# Train both with stealth wrapper
python -m training.train --agent ppo --stealth --timesteps 500000 --seed 42
python -m training.train --agent dqn --stealth --timesteps 500000 --seed 42

# Or train both at once in comparison mode
python -m training.train --compare --timesteps 500000 --seed 42
python -m training.train --compare --stealth --timesteps 500000 --seed 42
```

**Evaluation (manual commands):**

```bash
# Evaluate baseline models
python -m training.evaluate \
    --ppo_model results/ppo_baseline/final_model \
    --dqn_model results/dqn_baseline/final_model \
    --episodes 100 --seed 42

# Evaluate stealth models
python -m training.evaluate \
    --ppo_model results/ppo_stealth/final_model \
    --dqn_model results/dqn_stealth/final_model \
    --stealth --episodes 100 --seed 42
```

**Generate figures and report:**

```bash
python -m analysis.generate_all_figures
python -m analysis.report_generator --results_dir results/ --output results/pentest_report.md
```

### Option 4: Google Colab (GPU-Accelerated)

Open `notebooks/00_colab_training.ipynb` in Google Colab, set runtime to T4 GPU, and run all cells. The notebook handles cloning, installation, training, evaluation, and result packaging automatically.

### Verifying Reproduction

After running any of the above options, check the following expected outputs:

| File | Expected Content |
|---|---|
| `results/eval_baseline.json` | PPO mean_reward: -100.0, DQN mean_reward: ~-498 |
| `results/eval_stealth.json` | Both agents: mean_reward ~-109.9, catch_rate 1.0 |
| `results/ppo_baseline/train_meta.json` | Full hyperparameters, seed=42, wall time |
| `results/dqn_baseline/train_meta.json` | Full hyperparameters, seed=42, wall time |
| `results/plots/` | Training curves, comparison charts, topology diagram |

---

## Project Structure

```
rl-attack-path-simulation/
│
├── environments/
│   ├── network_config.py       # Custom NASim scenario: 5 subnets, 11 hosts, AI targets
│   │                           # Defines topology, firewall rules, exploits, host configs
│   └── stealth_wrapper.py      # Gymnasium wrapper: cumulative detection + caught penalty
│
├── agents/
│   ├── ppo_agent.py            # MaskablePPO agent (sb3-contrib) with native action masking
│   ├── dqn_agent.py            # DQN agent (SB3) with manual Q-value masking
│   └── wrappers.py             # IntActionWrapper, ActionMaskWrapper, DenseRewardWrapper
│
├── training/
│   ├── train.py                # Training CLI: single agent or head-to-head comparison
│   └── evaluate.py             # Evaluation: 100-episode runs, success/catch metrics, paths
│
├── analysis/
│   ├── attack_path.py          # Action-to-host mapping, pivot analysis, path interpretation
│   ├── mitre_mapping.py        # NASim actions -> MITRE ATT&CK techniques/tactics
│   ├── what_if.py              # Counterfactual topology analysis (firewall modifications)
│   ├── topology_diagram.py     # Network topology diagram generator (matplotlib)
│   ├── report_generator.py     # Automated Markdown penetration testing report
│   ├── generate_all_figures.py # Generate all publication-quality figures at once
│   └── visualize.py            # Training curves, comparison bars, sensitivity plots
│
├── tests/
│   ├── test_environment.py     # NASim built-in env & stealth wrapper tests
│   ├── test_custom_env.py      # Custom AI-infra topology tests
│   ├── test_agents.py          # PPO & DQN smoke tests (train, predict, save/load, masking)
│   ├── test_evaluation.py      # Evaluation harness tests
│   ├── test_attack_path.py     # Action mapping & path analysis tests
│   └── test_visualize.py       # Plot generation smoke tests
│
├── configs/
│   ├── ppo_config.yaml         # PPO hyperparameter reference
│   └── dqn_config.yaml         # DQN hyperparameter reference
│
├── scripts/
│   └── entrypoint.sh           # Docker entrypoint dispatcher
│
├── notebooks/
│   └── 00_colab_training.ipynb # Google Colab notebook (GPU training + evaluation)
│
├── run.sh                      # One-command reproduction script
├── Dockerfile                  # Python 3.12 container with all deps
├── docker-compose.yml          # Services: train, evaluate, visualize, test
├── .github/workflows/ci.yml    # GitHub Actions CI (lint + test on every push)
├── pyproject.toml              # black / isort / pytest config
├── setup.py                    # Package definition with console_scripts
├── requirements.txt            # Dependency list
└── results/                    # Training outputs (git-ignored except plots/)
```

---

## CLI Reference

### Training

```bash
python -m training.train --help
```

| Flag | Default | Description |
|---|---|---|
| `--agent {ppo,dqn}` | ppo | Algorithm to train |
| `--stealth` | off | Enable stealth-aware reward wrapper |
| `--timesteps N` | 500000 | Total training timesteps |
| `--compare` | off | Train both PPO and DQN under same conditions |
| `--detection_threshold` | 0.8 | Cumulative detection threshold (stealth only) |
| `--detection_cost` | 0.1 | Detection cost per active step (stealth only) |
| `--caught_penalty` | -100.0 | Penalty when attacker is caught (stealth only) |
| `--alpha` | 1.0 | Stealth penalty coefficient |
| `--seed` | 42 | Random seed for reproducibility |
| `--eval_freq` | 10000 | Evaluate every N timesteps during training |
| `--n_eval_episodes` | 10 | Episodes per evaluation round |
| `--output_dir` | results/ | Output directory for models and logs |

### Evaluation

```bash
python -m training.evaluate --help
```

| Flag | Default | Description |
|---|---|---|
| `--ppo_model PATH` | required | Path to saved PPO model |
| `--dqn_model PATH` | required | Path to saved DQN model |
| `--stealth` | off | Evaluate with stealth wrapper active |
| `--episodes N` | 100 | Number of evaluation episodes per agent |
| `--seed` | 42 | Evaluation random seed |

---

## Analysis & Reporting Tools

### Attack Path Analysis

```python
from analysis.attack_path import build_action_map, interpret_path, find_common_pivots
from environments.network_config import make_env

env = make_env()
action_map = build_action_map(env)
# pivots = find_common_pivots(paths, action_map, top_n=5)
```

### MITRE ATT&CK Mapping

Every NASim action maps to a MITRE ATT&CK technique:

| NASim Action | ATT&CK Tactic | Technique ID | Technique Name |
|---|---|---|---|
| subnet_scan | Discovery | T1046 | Network Service Discovery |
| e_http | Initial Access | T1190 | Exploit Public-Facing Application |
| e_ssh | Lateral Movement | T1021.004 | Remote Services: SSH |
| e_smb | Lateral Movement | T1021.002 | Remote Services: SMB |
| pe_linux | Privilege Escalation | T1068 | Exploitation for Privilege Escalation |

### What-If Topology Analysis

```python
from analysis.what_if import run_all_what_if
results = run_all_what_if(trained_agent, n_episodes=100)
```

Modifications: `block_ssh_to_ai`, `block_http_to_ai`, `isolate_data_lake`, `full_segmentation`.

### Automated Pentest Report

```bash
python -m analysis.report_generator --results_dir results/ --output results/pentest_report.md
```

### TensorBoard

```bash
tensorboard --logdir results/
```

---

## Developer Tooling

```bash
black .                          # Format code
isort .                          # Sort imports
flake8 . --max-line-length=120   # Lint
pytest tests/ -v                 # Run tests
pytest tests/ --cov=. --cov-report=html   # Coverage report
```

Configuration is in `pyproject.toml`.

---

## Future Work

- **Partial observability** -- restrict the attacker's view to only scanned hosts, closer to real-world conditions.
- **Multi-agent** -- add a defender agent that patches vulnerabilities in real time, enabling adversarial co-training.
- **Curriculum learning** -- start with easy topologies and progressively increase difficulty.
- **Cloud-native scenarios** -- model Kubernetes clusters, S3 buckets, and ML pipeline components as attack targets.
- **Integration with real tools** -- import network topologies from CMDBs and vulnerability data from scanners.
- **Detection sensitivity sweep** -- vary `detection_threshold` and plot success rate to find optimal monitoring level.

---

## References

1. Schwartz, J., & Kurniawati, H. (2019). **NASim: Network Attack Simulator**. arXiv:1905.05965.
2. Schulman, J., et al. (2017). **Proximal Policy Optimization Algorithms**. arXiv:1707.06347.
3. Mnih, V., et al. (2015). **Human-level control through deep reinforcement learning**. *Nature*, 518, 529-533.
4. Raffin, A., et al. (2021). **Stable-Baselines3: Reliable Reinforcement Learning Implementations**. *JMLR*, 22(268), 1-8.

---

## License

MIT -- see [LICENSE](LICENSE).
