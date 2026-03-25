"""Training package."""

from training.evaluate import evaluate_agent, evaluate_both_agents
from training.train import run_comparison, train_agent

__all__ = ["train_agent", "run_comparison", "evaluate_agent", "evaluate_both_agents"]
