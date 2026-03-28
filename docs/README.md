# Documentation

Extended documentation and design notes for the RL Attack-Path Simulation project.

**Author:** Syed Ali Turab — MMAI 845, Queen's University

## Available Documentation

See the top-level [README](../README.md) for:

- Architecture diagram and network topology
- MDP formulation (state, action, reward)
- Stealth reward wrapper design
- CLI reference and Docker usage
- Evaluation metrics and their definitions
- Full project structure

## Design Notes

### Stealth Reward Wrapper

The stealth wrapper (`environments/stealth_wrapper.py`) models a defender's
Security Operations Center (SOC). Detection accumulates **only** on active
steps (scan, exploit, privilege escalation) — idle actions do not raise the
alarm. The reward penalty is applied consistently with detection accumulation:

```
Active step:  r' = r_env - alpha * detection_cost_per_step
Idle step:    r' = r_env  (passed through unmodified)
Caught:       r' += caught_penalty  (episode terminates)
```

### Success Rate Definition

Success rate measures whether the agent **actually reached AI infrastructure**
(total episode reward ≥ 100), not merely whether it avoided detection. This
ensures the metric is meaningful in both baseline and stealth modes.

### Attack Path Analysis

The `analysis/attack_path.py` module maps NASim's flat action indices back to
human-readable `(subnet, host, action_type)` tuples. This enables security
analysis: identifying which hosts are common pivot points, what the typical
attack chain looks like, and where to place network controls.
