# Using AI to Secure AI: RL for Attack Path Simulation Against Enterprise AI Infrastructure

> **MMAI 845 – Reinforcement Learning | Syed Ali Turab**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.3%2B-orange)](https://stable-baselines3.readthedocs.io/)
[![NASim](https://img.shields.io/badge/NASim-0.10%2B-green)](https://networkattacksimulator.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

Enterprises are deploying AI infrastructure — LLM servers, vector databases, and model repositories — into corporate networks that were not designed to protect these assets. Traditional penetration tests are expensive and infrequent; static vulnerability scanners do not model multi-step attacker behaviour.

This project addresses that gap using **Reinforcement Learning**. An RL agent acts as an attacker navigating a simulated corporate network, learning efficient paths to high-value AI infrastructure. Security teams can use the outputs to:

- Identify which hosts serve as **stepping stones** to AI assets
- Evaluate whether a proposed **network change** reduces attack success
- Compare attacker behaviour under **different detection regimes**

### Research Questions

1. Can PPO and DQN learn effective multi-hop attack paths to AI infrastructure in a simulated enterprise network?
2. How does a stealth-aware reward signal change the attack strategy learned by each algorithm?
3. Which algorithm produces faster convergence and higher attack success rate?

---

## Project Status

| Component | Status |
|---|---|
| NASim AI-infrastructure network topology | Implemented |
| Stealth-aware reward wrapper | Implemented |
| PPO agent (Stable-Baselines3) | Implemented |
| DQN agent (Stable-Baselines3) | Implemented |
| Training & evaluation CLI harness | Implemented |
| Attack path interpretation & analysis | Implemented |
| Visualisation / plotting utilities | Implemented |
| Docker containerised training pipeline | Implemented |
| Unit & integration tests | Implemented |
| GitHub Actions CI | Implemented |
| Full training runs + saved models | In Progress |
| Exploratory notebooks | Planned |

---

## Architecture

```
Subnet 0 (Attacker)
    │
    ▼
Subnet 1 — DMZ (Web, Email, DNS)
    │
    ▼
Subnet 2 — Corporate LAN (Workstations, AD, Internal Services)
    │
    ▼
Subnet 3 — AI Infrastructure  ◄─── HIGH VALUE
    │   LLM API Server (reward: 200)
    │   Vector DB       (reward: 200)
    │   Model Repo      (reward: 150)
    ▼
Subnet 4 — Data Lake           ◄─── HIGHEST VALUE
        Training Data           (reward: 300)
```

The attacker starts at the internet boundary and must pivot through firewalled subnets using scan → exploit → privilege-escalation chains. Firewall rules restrict lateral movement: the LLM API Server only accepts HTTP from one corporate host and SSH from another, forcing the agent to discover and exploit specific pivot chains.

---

## Project Structure

```
rl-attack-path-simulation/
│
├── environments/
│   ├── network_config.py       # NASim YAML scenario: 5 subnets, 12 hosts, AI-infra targets
│   └── stealth_wrapper.py      # Gymnasium wrapper: detection tracking + reward shaping
│
├── agents/
│   ├── ppo_agent.py            # PPO attacker (Stable-Baselines3)
│   ├── dqn_agent.py            # DQN attacker (Stable-Baselines3)
│   └── wrappers.py             # Shared action wrapper (NASim int coercion)
│
├── training/
│   ├── train.py                # Training entry point (CLI + API, full hyperparams)
│   └── evaluate.py             # Evaluation: goal-based success rate, per-episode metrics
│
├── analysis/
│   ├── attack_path.py          # Action→host mapping, pivot analysis, path interpretation
│   └── visualize.py            # Training curves, comparison bars, sensitivity plots
│
├── tests/
│   ├── test_environment.py     # NASim built-in env & stealth wrapper tests
│   ├── test_custom_env.py      # Custom AI-infra topology tests
│   ├── test_agents.py          # PPO & DQN smoke tests (train, predict, save/load)
│   ├── test_evaluation.py      # Evaluation harness tests
│   ├── test_attack_path.py     # Action mapping & path analysis tests
│   └── test_visualize.py       # Plot generation smoke tests
│
├── scripts/
│   └── entrypoint.sh           # Docker entrypoint dispatcher
│
├── Dockerfile                  # Python 3.12 container with all deps
├── docker-compose.yml          # Services: train, evaluate, visualize, test
├── .dockerignore
├── .github/workflows/ci.yml    # GitHub Actions CI (lint + test)
├── pyproject.toml              # black / isort / pytest config
├── setup.py                    # Package definition with console_scripts
├── results/                    # Saved models, logs, plots (git-ignored)
└── README.md
```

---

## Methodology

### Environment: NASim

[Network Attack Simulator (NASim)](https://networkattacksimulator.readthedocs.io/) provides a Gymnasium-compatible discrete MDP over a parameterised corporate network.

| Component | Description |
|---|---|
| **State** | Flat vector: per-host discovery, access level, compromise status |
| **Actions** | Discrete: `(host, action_type)` pairs — scan, exploit, privilege escalation |
| **Rewards** | +value on **first** compromise of each host; penalty for invalid actions |
| **Preconditions** | NASim enforces action preconditions natively; invalid actions return penalty with no state change |

### Reward Variants

| Variant | Reward Signal |
|---|---|
| **Baseline** | Raw NASim rewards only |
| **Stealth-aware** | On active steps: `r' = r_env − α × detection_cost`; idle steps pass through unmodified. Episode terminates if cumulative detection ≥ threshold |

The stealth reward models a sophisticated adversary who must balance progress against triggering the defender's detection controls. Detection only accumulates on active steps (scan, exploit, priv-esc), not idle actions.

### Agents

| Agent | Algorithm | Key Hyperparameters |
|---|---|---|
| PPO | Proximal Policy Optimization | lr=3e-4, n_steps=2048, clip=0.2, ent=0.01, γ=0.99 |
| DQN | Deep Q-Network | lr=1e-4, buffer=100k, ε: 1.0→0.05 over 20% of training, γ=0.99 |

Both use a 2-layer MLP (256×256) and are trained with identical environments, seeds, and evaluation protocols for a fair comparison.

### Evaluation Metrics

| Metric | Definition |
|---|---|
| **Success Rate** | Fraction of episodes where total reward ≥ 100 (agent reached AI infrastructure) |
| **Catch Rate** | Fraction of episodes terminated by detection (stealth mode only) |
| **Mean Reward** | Average episode return across evaluation episodes |
| **Mean Steps** | Average episode length |

---

## Quickstart

### Option A: Docker (Recommended)

No local Python setup needed — all dependencies are baked into the container.

```bash
# Clone the repo
git clone https://github.com/turaab97/rl-attack-path-simulation.git
cd rl-attack-path-simulation

# Train baseline PPO + DQN comparison (500k steps each)
docker compose run train

# Train stealth mode comparison
docker compose run train-stealth

# Evaluate saved models (100 episodes each)
docker compose run evaluate

# Generate plots
docker compose run visualize

# Run test suite
docker compose run test
```

Results are persisted to `./results/` on the host via a bind mount.

### Option B: Local Installation

```bash
# Requires Python 3.10–3.12 (PyTorch does not support 3.13 yet)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install as editable package with dev tools
pip install -e ".[dev]"
```

### Train a Single Agent

```bash
# PPO baseline, 500k steps
python -m training.train --agent ppo --timesteps 500000

# DQN with stealth reward and custom detection parameters
python -m training.train --agent dqn --stealth \
    --timesteps 500000 \
    --detection_threshold 0.8 \
    --detection_cost 0.1 \
    --caught_penalty -100.0 \
    --alpha 1.0

# Compare both agents under the same conditions
python -m training.train --compare --timesteps 300000
```

### Evaluate Trained Agents

```bash
python -m training.evaluate \
    --ppo_model results/ppo_baseline/final_model \
    --dqn_model results/dqn_baseline/final_model \
    --episodes 100
```

### Generate Visualisations

```bash
python -m analysis.visualize --results_dir results/ --output_dir results/plots
```

### Monitor Training with TensorBoard

```bash
tensorboard --logdir results/
```

### Run Tests

```bash
pytest tests/ -v
pytest tests/ --cov=. --cov-report=html
```

---

## Stealth-Aware Reward Wrapper

```python
from environments.stealth_wrapper import make_stealth_env
from environments.network_config import make_env

base_env = make_env()  # Custom AI-infra topology
env = make_stealth_env(
    base_env,
    detection_threshold=0.8,      # Episode ends if cumulative risk >= this
    detection_cost_per_step=0.1,   # Risk added per active step
    caught_penalty=-100.0,         # Terminal reward when caught
    alpha=1.0,                     # Detection cost multiplier
)

obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print(info["cumulative_detection"])   # Current detection risk
print(info["caught"])                 # Whether attacker was caught
print(env.stealth_budget_remaining)   # Budget remaining (1.0 → 0.0)
```

---

## Attack Path Analysis

```python
from analysis.attack_path import build_action_map, interpret_path, find_common_pivots
from environments.network_config import make_env

env = make_env()
action_map = build_action_map(env)

# After running evaluation episodes:
# paths = [episode["path"] for episode in eval_results["per_episode"]]
# pivots = find_common_pivots(paths, action_map, top_n=5)
# → [("Web Server", 87), ("Dev Server", 64), ("LLM API Server", 52), ...]
```

This directly answers the security question: **which hosts serve as stepping stones to AI infrastructure?**

---

## Reproducibility

| Item | Value |
|---|---|
| Python version | 3.10–3.12 (Docker uses 3.12) |
| Key dependencies | See `setup.py` |
| Default random seed | `42` (pass `--seed` to override) |
| Hardware | CPU sufficient; GPU speeds up longer training runs |

All hyperparameters, stealth parameters, and seeds are saved to `train_meta.json` after each training run for full reproducibility.

```bash
python -m training.train --agent ppo --timesteps 500000 --seed 42
```

---

## Output Artifacts

After a training run, `results/` contains:

```
results/
├── ppo_baseline/
│   ├── final_model.zip          # Saved PPO weights
│   ├── best_model.zip           # Best checkpoint by eval reward
│   ├── train_meta.json          # Full hyperparameters, seed, stealth params, wall time
│   └── tensorboard/             # TensorBoard logs
├── dqn_baseline/
│   └── ...                      # Same structure
├── eval_results.json            # Aggregate evaluation metrics (PPO vs DQN)
└── plots/
    ├── training_curves.png      # Episode reward / length over training
    ├── comparison_bar.png       # Bar chart: key metrics side-by-side
    ├── attack_path.png          # Action sequence timeline
    └── detection_sensitivity.png  # Success rate vs detection threshold
```

> Trained models and plots are `.gitignore`d. Run the training pipeline locally or via Docker to generate them.

---

## CLI Reference

```bash
# Training
python -m training.train --help
  --agent {ppo,dqn}             Algorithm (default: ppo)
  --stealth                     Enable stealth wrapper
  --timesteps N                 Training budget (default: 500000)
  --compare                     Train both PPO and DQN
  --detection_threshold FLOAT   Detection threshold (default: 0.8)
  --detection_cost FLOAT        Cost per active step (default: 0.1)
  --caught_penalty FLOAT        Caught penalty (default: -100.0)
  --alpha FLOAT                 Penalty coefficient (default: 1.0)
  --seed INT                    Random seed (default: 42)
  --eval_freq INT               Eval every N steps (default: 10000)
  --n_eval_episodes INT         Episodes per eval (default: 10)
  --output_dir PATH             Output directory (default: results/)

# Evaluation
python -m training.evaluate --help
  --ppo_model PATH              PPO model path (required)
  --dqn_model PATH              DQN model path (required)
  --stealth                     Evaluate with stealth wrapper
  --episodes INT                Eval episodes (default: 100)
  --detection_threshold FLOAT   (default: 0.8)
  --detection_cost FLOAT        (default: 0.1)
  --caught_penalty FLOAT        (default: -100.0)
  --alpha FLOAT                 (default: 1.0)
```

---

## Developer Tooling

```bash
black .          # Format code
isort .          # Sort imports
flake8 .         # Lint
pytest tests/ -v # Run tests
```

Configuration lives in `pyproject.toml`.

---

## Future Work

- **Detection sensitivity sweep** — vary `detection_threshold` and plot success rate for both agents to find the optimal monitoring level.
- **Richer NASim scenarios** — model lateral movement through cloud-native AI pipelines (Kubernetes, S3, MLflow serving).
- **Reward shaping experiments** — curriculum learning: start with a low detection threshold and anneal upward as the agent improves.
- **Multi-agent extension** — add a defender agent that dynamically patches vulnerabilities, enabling adversarial co-training.
- **Partial observability** — restrict the attacker's observation to only discovered hosts, closer to real-world conditions.
- **Cloud deployment** — run as a scheduled container job (e.g. AWS ECS, Azure Container Instances) triggered by network configuration changes for continuous attack-path assessment.

---

## References

1. Schwartz, J., & Kurniawati, H. (2019). **NASim: Network Attack Simulator**. arXiv:1905.05965.
2. Schulman, J., et al. (2017). **Proximal Policy Optimization Algorithms**. arXiv:1707.06347.
3. Mnih, V., et al. (2015). **Human-level control through deep reinforcement learning**. *Nature*, 518, 529–533.
4. Raffin, A., et al. (2021). **Stable-Baselines3: Reliable Reinforcement Learning Implementations**. *JMLR*, 22(268), 1–8.

---

## Author

**Syed Ali Turab** — [info@turab.sh](mailto:info@turab.sh)

MMAI 845 – Reinforcement Learning, Queen's University

---

## License

MIT – see [LICENSE](LICENSE).
