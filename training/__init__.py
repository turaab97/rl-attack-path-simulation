"""Training package."""
from training.train import train_agent, run_comparison
from training.evaluate import evaluate_agent, evaluate_both_agents

__all__ = ["train_agent", "run_comparison", "evaluate_agent", "evaluate_both_agents"]
