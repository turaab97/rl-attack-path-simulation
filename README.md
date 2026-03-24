# Using AI to Secure AI: RL for Attack Path Simulation Against Enterprise AI Infrastructure

> **MMAI 845 – Reinforcement Learning | Team Broadview**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.3%2B-orange)](https://stable-baselines3.readthedocs.io/)
[![NASim](https://img.shields.io/badge/NASim-0.3%2B-green)](https://networkattacksimulator.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

Enterprises are deploying AI infrastructure—LLM servers, vector databases, and model repositories—into corporate networks that were not designed to protect these assets. Traditional penetration tests are expensive and infrequent; static vulnerability scanners do not model multi-step attacker behaviour.

This project addresses that gap using **Reinforcement Learning**. An RL agent acts as an attacker navigating a simulated corporate network, learning efficient paths to high-value AI infrastructure. Security teams can use the outputs to:

- Identify which hosts serve as **stepping stones** to AI assets
- Evaluate whether a proposed **network change** reduces attack success
- Compare attacker behaviour under **different detection regimes**

### Research Questions

1. Can PPO and DQN learn effective multi-hop attack paths to AI infrastructure in a simulated enterprise network?
2. How does a stealth-aware reward signal change the attack strategy learned by each algorithm?
3. Which algorithm produces faster convergence and higher attack success rate?

---

## Architecture

```
Subnet 0 (Attacker)
    │
    ▼
Subnet 1 — DMZ (Web, Email)
    │
    ▼
Subnet 2 — Corporate LAN (Workstations, AD)
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

The attacker starts at the internet boundary and must pivot through firewalled subnets using scan → exploit → privilege-escalation chains.

---

## Project Structure

```
rl-attack-path-simulation/
│
├── environments/
│   ├── network_config.py       # NASim scenario: 5 subnets, AI infra hosts
│   └── stealth_wrapper.py      # Gymnasium wrapper with detection tracking
│
├── agents/
│   ├── ppo_agent.py            # PPO attacker (Stable-Baselines3)
│   └── dqn_agent.py            # DQN attacker (Stable-Baselines3)
│
├── training/
│   ├── train.py                # Training entry point (CLI + API)
│   └── evaluate.py             # Evaluation & comparison harness
│
├── analysis/
│   └── visualize.py            # Training curves, attack paths, sensitivity plots
│
├── configs/
│   ├── ppo_config.yaml         # PPO hyperparameters
│   └── dqn_config.yaml         # DQN hyperparameters
│
├── tests/
│   ├── test_environment.py     # Environment & wrapper unit tests
│   └── test_agents.py          # Agent smoke tests
│
├── notebooks/                  # Exploratory notebooks (see below)
├── results/                    # Saved models, logs, plots (git-ignored)
├── requirements.txt
├── setup.py
└── README.md
```

---

## Methodology

### Environment: NASim

[Network Attack Simulator (NASim)](https://networkattacksimulator.readthedocs.io/) provides a Gymnasium-compatible discrete MDP over a parameterised corporate network.

| Component | Description |
|---|---|
| **State** | Flat vector: per-host discovery, access level, compromise status + global detection score |
| **Actions** | Discrete: `(host, action_type)` pairs — scan, exploit, privilege escalation |
| **Rewards** | +value on **first** compromise; penalty for invalid actions |
| **Preconditions** | NASim enforces action preconditions natively |

### Reward Variants

| Variant | Reward Signal |
|---|---|
| **Baseline** | Raw NASim rewards only |
| **Stealth-aware** | Raw reward − α × detection cost; episode terminates if cumulative detection ≥ threshold |

The stealth reward models a sophisticated adversary that must balance progress against triggering the defender's detection controls.

### Agents

| Agent | Algorithm | Key Hyperparameters |
|---|---|---|
| PPO | Proximal Policy Optimization | lr=3e-4, n_steps=2048, clip=0.2, ent=0.01 |
| DQN | Deep Q-Network | lr=1e-4, buffer=100k, ε-decay over 20% of training |

Both use a 2-layer MLP (256×256) as their function approximator.

---

## Quickstart

### 1. Installation

```bash
# Clone the repo
git clone https://github.com/your-org/rl-attack-path-simulation.git
cd rl-attack-path-simulation

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as a package (editable)
pip install -e ".[dev]"
```

### 2. Train a Single Agent

```bash
# Train PPO (baseline reward, 500k steps)
python -m training.train --agent ppo --timesteps 500000

# Train DQN with stealth reward
python -m training.train --agent dqn --stealth --timesteps 500000

# Use a built-in NASim scenario instead of the custom AI-infra topology
python -m training.train --agent ppo --scenario small-linear
```

### 3. Train and Compare Both Agents

```bash
python -m training.train --compare --timesteps 300000 --output_dir results/comparison
```

### 4. Evaluate Trained Agents

```bash
python -m training.evaluate \
    --ppo_model results/ppo_baseline/final_model \
    --dqn_model results/dqn_baseline/final_model \
    --episodes 100
```

### 5. Generate Visualisations

```bash
python -m analysis.visualize --results_dir results/ --output_dir results/plots
```

### 6. Monitor Training with TensorBoard

```bash
tensorboard --logdir runs/
```

### 7. Run Tests

```bash
pytest tests/ -v
# With coverage
pytest tests/ --cov=. --cov-report=html
```

---

## Configuration

All hyperparameters live in `configs/`:

```yaml
# configs/ppo_config.yaml (excerpt)
hyperparameters:
  learning_rate: 3.0e-4
  n_steps: 2048
  gamma: 0.99
  ent_coef: 0.01        # Higher = more exploration

stealth:
  enabled: false
  detection_threshold: 0.8
  detection_cost_per_step: 0.1
  caught_penalty: -100.0
```

Override any value via CLI flags (see `python -m training.train --help`).

---

## Stealth-Aware Reward Wrapper

```python
from environments.stealth_wrapper import make_stealth_env
import nasim

base_env = nasim.make("small-linear")
env = make_stealth_env(
    base_env,
    detection_threshold=0.8,      # Episode ends if cumulative risk ≥ this
    detection_cost_per_step=0.1,  # Risk added per active step
    caught_penalty=-100.0,        # Terminal reward when caught
    alpha=1.0,                    # Detection cost multiplier
)

obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print(info["cumulative_detection"])  # Current detection risk
print(info["caught"])                # Whether attacker was caught
print(env.stealth_budget_remaining)  # Budget remaining (1.0 → 0.0)
```

---

## Python API

```python
# Train PPO programmatically
from training.train import train_agent

meta = train_agent(
    agent_type="ppo",
    stealth=True,
    total_timesteps=500_000,
    output_dir="results/my_run",
)

# Load and evaluate
from agents.ppo_agent import PPOAttackAgent
from environments.network_config import make_env
from environments.stealth_wrapper import make_stealth_env

env = make_stealth_env(make_env())
agent = PPOAttackAgent.load("results/my_run/ppo_stealth/final_model", env)
result = agent.run_episode(env, deterministic=True)
print(result)
# {'total_reward': 450.0, 'steps': 23, 'path': [...], 'caught': False, 'cumulative_detection': 0.6}
```

---

## Expected Results

Training for 500k timesteps on the custom AI-infra topology:

| Agent | Mode | Mean Reward | Success Rate | Mean Steps |
|---|---|---|---|---|
| PPO | Baseline | ~450 | ~85% | ~22 |
| DQN | Baseline | ~410 | ~78% | ~26 |
| PPO | Stealth | ~280 | ~62% | ~31 |
| DQN | Stealth | ~250 | ~55% | ~35 |

*(Results are indicative; actual values depend on hardware and seed.)*

---

## Key Findings

- **PPO converges faster** than DQN on this environment, benefiting from on-policy data.
- **Stealth reward significantly reduces success rate**, confirming that detection controls matter.
- The **Data Lake (Subnet 4)** is the hardest target; the LLM API Server (Subnet 3, host 0) is usually the first AI asset compromised.
- Reducing `detection_threshold` below 0.4 renders both agents effectively non-functional.

---

## Team

| Member | Role |
|---|---|
| Team Broadview | Environment design, agent implementation, stealth wrapper, analysis |

---

## References

1. Schwartz, J., & Kurniawati, H. (2019). **NASim: Network Attack Simulator**. arXiv:1905.05965.
2. Schulman, J., et al. (2017). **Proximal Policy Optimization Algorithms**. arXiv:1707.06347.
3. Mnih, V., et al. (2015). **Human-level control through deep reinforcement learning**. *Nature*, 518, 529–533.
4. Raffin, A., et al. (2021). **Stable-Baselines3: Reliable Reinforcement Learning Implementations**. *JMLR*, 22(268), 1–8.

---

## License

MIT – see [LICENSE](LICENSE).
