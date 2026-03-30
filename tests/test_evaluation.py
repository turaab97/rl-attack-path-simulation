"""
test_evaluation.py
------------------
Tests for the evaluation harness (training/evaluate.py).

Author: Syed Ali Turab
Course: MMAI 845 – Reinforcement Learning
"""

import numpy as np


class TestEvaluateAgent:
    """Test evaluate_agent with a randomly initialised agent (no real training)."""

    def test_evaluate_returns_expected_keys(self):
        import nasim

        from agents.ppo_agent import PPOAttackAgent
        from training.evaluate import evaluate_agent

        env = nasim.make_benchmark("small-linear")
        agent = PPOAttackAgent(env=env, tensorboard_log=None)

        results = evaluate_agent(agent, env, n_episodes=3, deterministic=True)

        assert "n_episodes" in results
        assert "mean_reward" in results
        assert "std_reward" in results
        assert "mean_steps" in results
        assert "catch_rate" in results
        assert "success_rate" in results
        assert "mean_ai_hosts_reached" in results
        assert "ai_host_reach_rate" in results
        assert "ai_host_reach_counts" in results
        assert "successful_target_paths" in results
        assert "per_episode" in results
        assert results["n_episodes"] == 3
        assert len(results["per_episode"]) == 3
        env.close()

    def test_success_rate_is_between_zero_and_one(self):
        import nasim

        from agents.dqn_agent import DQNAttackAgent
        from training.evaluate import evaluate_agent

        env = nasim.make_benchmark("small-linear")
        agent = DQNAttackAgent(env=env, tensorboard_log=None, learning_starts=10)

        results = evaluate_agent(agent, env, n_episodes=5, deterministic=True)
        assert 0.0 <= results["success_rate"] <= 1.0
        assert 0.0 <= results["catch_rate"] <= 1.0
        assert 0.0 <= results["ai_host_reach_rate"] <= 1.0
        env.close()

    def test_rewards_are_finite(self):
        import nasim

        from agents.ppo_agent import PPOAttackAgent
        from training.evaluate import evaluate_agent

        env = nasim.make_benchmark("small-linear")
        agent = PPOAttackAgent(env=env, tensorboard_log=None)

        results = evaluate_agent(agent, env, n_episodes=2, deterministic=True)
        assert np.isfinite(results["mean_reward"])
        assert np.isfinite(results["std_reward"])
        env.close()

    def test_per_episode_has_path(self):
        import nasim

        from agents.ppo_agent import PPOAttackAgent
        from training.evaluate import evaluate_agent

        env = nasim.make_benchmark("small-linear")
        agent = PPOAttackAgent(env=env, tensorboard_log=None)

        results = evaluate_agent(agent, env, n_episodes=1, deterministic=True)
        ep = results["per_episode"][0]
        assert "total_reward" in ep
        assert "steps" in ep
        assert "path" in ep
        assert "target_path" in ep
        assert "ai_hosts_reached" in ep
        assert isinstance(ep["path"], list)
        assert isinstance(ep["target_path"], list)
        assert isinstance(ep["ai_hosts_reached"], list)
        env.close()
