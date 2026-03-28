"""
test_agents.py
--------------
Smoke tests for PPO and DQN agents.
Trains for a tiny number of timesteps to verify the pipeline works end-to-end.

Author: Syed Ali Turab
Course: MMAI 845 – Reinforcement Learning
"""

import os

import nasim
import pytest

SMOKE_TIMESTEPS = 2_000


@pytest.fixture(scope="module")
def small_env():
    env = nasim.make_benchmark("small-linear")
    yield env
    env.close()


class TestPPOAgent:
    def test_ppo_initialises(self):
        from agents.ppo_agent import PPOAttackAgent

        env = nasim.make_benchmark("small-linear")
        agent = PPOAttackAgent(env=env, tensorboard_log=None)
        assert agent.model is not None
        env.close()

    def test_ppo_trains_smoke(self):
        from agents.ppo_agent import PPOAttackAgent

        env = nasim.make_benchmark("small-linear")
        agent = PPOAttackAgent(env=env, tensorboard_log=None)
        agent.train(total_timesteps=SMOKE_TIMESTEPS)
        env.close()

    def test_ppo_predict_returns_valid_action(self):
        from agents.ppo_agent import PPOAttackAgent

        env = nasim.make_benchmark("small-linear")
        agent = PPOAttackAgent(env=env, tensorboard_log=None)
        obs, _ = env.reset()
        action = agent.predict(obs)
        assert 0 <= action < env.action_space.n
        env.close()

    def test_ppo_save_and_load(self, tmp_path):
        from agents.ppo_agent import PPOAttackAgent

        env = nasim.make_benchmark("small-linear")
        agent = PPOAttackAgent(env=env, tensorboard_log=None)
        save_path = str(tmp_path / "ppo_test")
        agent.save(save_path)
        assert os.path.exists(save_path + ".zip")

        load_env = nasim.make_benchmark("small-linear")
        loaded = PPOAttackAgent.load(save_path, load_env)
        obs, _ = load_env.reset()
        action = loaded.predict(obs)
        assert 0 <= action < load_env.action_space.n
        env.close()
        load_env.close()

    def test_ppo_run_episode(self):
        from agents.ppo_agent import PPOAttackAgent

        env = nasim.make_benchmark("small-linear")
        agent = PPOAttackAgent(env=env, tensorboard_log=None)
        result = agent.run_episode(env)
        assert "total_reward" in result
        assert "steps" in result
        assert "path" in result
        assert isinstance(result["path"], list)
        assert result["steps"] > 0
        env.close()


class TestDQNAgent:
    def test_dqn_initialises(self):
        from agents.dqn_agent import DQNAttackAgent

        env = nasim.make_benchmark("small-linear")
        agent = DQNAttackAgent(env=env, tensorboard_log=None)
        assert agent.model is not None
        env.close()

    def test_dqn_trains_smoke(self):
        from agents.dqn_agent import DQNAttackAgent

        env = nasim.make_benchmark("small-linear")
        agent = DQNAttackAgent(env=env, tensorboard_log=None, learning_starts=100)
        agent.train(total_timesteps=SMOKE_TIMESTEPS)
        env.close()

    def test_dqn_predict_returns_valid_action(self):
        from agents.dqn_agent import DQNAttackAgent

        env = nasim.make_benchmark("small-linear")
        agent = DQNAttackAgent(env=env, tensorboard_log=None)
        obs, _ = env.reset()
        action = agent.predict(obs)
        assert 0 <= action < env.action_space.n
        env.close()

    def test_dqn_save_and_load(self, tmp_path):
        from agents.dqn_agent import DQNAttackAgent

        env = nasim.make_benchmark("small-linear")
        agent = DQNAttackAgent(env=env, tensorboard_log=None)
        save_path = str(tmp_path / "dqn_test")
        agent.save(save_path)
        assert os.path.exists(save_path + ".zip")

        load_env = nasim.make_benchmark("small-linear")
        loaded = DQNAttackAgent.load(save_path, load_env)
        obs, _ = load_env.reset()
        action = loaded.predict(obs)
        assert 0 <= action < load_env.action_space.n
        env.close()
        load_env.close()

    def test_dqn_run_episode(self):
        from agents.dqn_agent import DQNAttackAgent

        train_env = nasim.make_benchmark("small-linear")
        eval_env = nasim.make_benchmark("small-linear")
        agent = DQNAttackAgent(env=train_env, tensorboard_log=None, learning_starts=100)
        agent.train(total_timesteps=SMOKE_TIMESTEPS)
        result = agent.run_episode(eval_env)
        assert "total_reward" in result
        assert "steps" in result
        assert isinstance(result["path"], list)
        train_env.close()
        eval_env.close()


class TestSharedWrapper:
    def test_int_action_wrapper_import(self):
        """Verify the shared wrapper can be imported from agents.wrappers."""
        from agents.wrappers import IntActionWrapper

        assert IntActionWrapper is not None
