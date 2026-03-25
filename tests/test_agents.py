"""
test_agents.py
--------------
Smoke tests for PPO and DQN agents.
Trains for a tiny number of timesteps to verify the pipeline works end-to-end.
"""

import nasim
import pytest

SMOKE_TIMESTEPS = 2_000  # Very short: just enough to verify no crashes


@pytest.fixture(scope="module")
def small_env():
    env = nasim.make("small-linear")
    yield env
    env.close()


class TestPPOAgent:
    def test_ppo_initialises(self, small_env):
        from agents.ppo_agent import PPOAttackAgent

        agent = PPOAttackAgent(env=small_env, tensorboard_log=None)
        assert agent.model is not None

    def test_ppo_trains_smoke(self, tmp_path, small_env):
        import nasim

        from agents.ppo_agent import PPOAttackAgent

        env = nasim.make("small-linear")
        agent = PPOAttackAgent(env=env, tensorboard_log=None)
        agent.train(total_timesteps=SMOKE_TIMESTEPS)
        env.close()

    def test_ppo_predict_returns_valid_action(self, small_env):
        import nasim

        from agents.ppo_agent import PPOAttackAgent

        env = nasim.make("small-linear")
        agent = PPOAttackAgent(env=env, tensorboard_log=None)
        obs, _ = env.reset()
        action = agent.predict(obs)
        assert 0 <= action < env.action_space.n
        env.close()

    def test_ppo_save_load(self, tmp_path, small_env):
        import nasim

        from agents.ppo_agent import PPOAttackAgent

        env = nasim.make("small-linear")
        agent = PPOAttackAgent(env=env, tensorboard_log=None)
        save_path = str(tmp_path / "ppo_test")
        agent.save(save_path)
        import os

        assert os.path.exists(save_path + ".zip")
        env.close()


class TestDQNAgent:
    def test_dqn_initialises(self, small_env):
        import nasim

        from agents.dqn_agent import DQNAttackAgent

        env = nasim.make("small-linear")
        agent = DQNAttackAgent(env=env, tensorboard_log=None)
        assert agent.model is not None
        env.close()

    def test_dqn_trains_smoke(self, tmp_path, small_env):
        import nasim

        from agents.dqn_agent import DQNAttackAgent

        env = nasim.make("small-linear")
        agent = DQNAttackAgent(env=env, tensorboard_log=None, learning_starts=100)
        agent.train(total_timesteps=SMOKE_TIMESTEPS)
        env.close()

    def test_dqn_predict_returns_valid_action(self, small_env):
        import nasim

        from agents.dqn_agent import DQNAttackAgent

        env = nasim.make("small-linear")
        agent = DQNAttackAgent(env=env, tensorboard_log=None)
        obs, _ = env.reset()
        action = agent.predict(obs)
        assert 0 <= action < env.action_space.n
        env.close()

    def test_dqn_run_episode(self, small_env):
        import nasim

        from agents.dqn_agent import DQNAttackAgent

        train_env = nasim.make("small-linear")
        eval_env = nasim.make("small-linear")
        agent = DQNAttackAgent(env=train_env, tensorboard_log=None, learning_starts=100)
        agent.train(total_timesteps=SMOKE_TIMESTEPS)
        result = agent.run_episode(eval_env)
        assert "total_reward" in result
        assert "steps" in result
        assert isinstance(result["path"], list)
        train_env.close()
        eval_env.close()
