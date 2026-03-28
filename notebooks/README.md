# Notebooks

Exploratory Jupyter notebooks for the RL Attack-Path Simulation project.

**Author:** Syed Ali Turab — MMAI 845, Queen's University

## Planned Notebooks

- `01_environment_exploration.ipynb` — step through the NASim AI-infra topology interactively, inspect obs/action spaces, verify PPO/DQN compatibility
- `02_training_analysis.ipynb` — load monitor CSVs and reproduce training curves with multi-seed error bars
- `03_attack_path_replay.ipynb` — replay a saved agent episode step-by-step, map actions to host names, visualise the pivot chain
- `04_sensitivity_sweep.ipynb` — vary `detection_threshold` and plot success rate vs stealth for both agents

## Prerequisites

Train models first:

```bash
# Via Docker
docker compose run train
docker compose run train-stealth

# Or locally
python -m training.train --compare --timesteps 500000
python -m training.train --compare --stealth --timesteps 500000
```

## Launch

```bash
pip install jupyterlab
jupyter lab notebooks/
```
