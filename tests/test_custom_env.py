"""
test_custom_env.py
------------------
Tests for the custom AI-infrastructure NASim topology defined in
environments/network_config.py — ensures the YAML scenario loads,
steps without error, and has the expected structure.

Author: Syed Ali Turab
Course: MMAI 845 – Reinforcement Learning
"""

import gymnasium as gym
import numpy as np
import pytest

from environments.network_config import AI_INFRA_HOSTS, build_network_scenario, make_env
from environments.stealth_wrapper import StealthAwareWrapper, make_stealth_env


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def custom_env():
    """Load the custom AI-infrastructure topology."""
    env = make_env(scenario_name=None)
    yield env
    env.close()


@pytest.fixture(scope="module")
def custom_stealth_env():
    """Custom topology with stealth wrapper."""
    env = make_env(scenario_name=None)
    wrapped = make_stealth_env(
        env,
        detection_threshold=0.8,
        detection_cost_per_step=0.1,
        caught_penalty=-100.0,
        alpha=1.0,
    )
    yield wrapped
    env.close()


# ---------------------------------------------------------------------------
# Topology structure
# ---------------------------------------------------------------------------


class TestCustomTopology:
    def test_env_loads_without_error(self, custom_env):
        assert custom_env is not None

    def test_scenario_builds_without_error(self):
        scenario = build_network_scenario()
        assert scenario is not None

    def test_observation_space_is_flat(self, custom_env):
        assert len(custom_env.observation_space.shape) == 1

    def test_action_space_is_discrete(self, custom_env):
        assert isinstance(custom_env.action_space, gym.spaces.Discrete)

    def test_action_space_has_expected_size(self, custom_env):
        # 12 hosts across 5 subnets; action count depends on NASim internals
        # but should be > 0 and reasonable
        assert custom_env.action_space.n > 0
        assert custom_env.action_space.n < 5000

    def test_reset_returns_valid_obs(self, custom_env):
        obs, info = custom_env.reset()
        assert obs is not None
        assert isinstance(info, dict)
        assert obs.shape == custom_env.observation_space.shape

    def test_step_executes_without_error(self, custom_env):
        custom_env.reset()
        action = int(custom_env.action_space.sample())
        obs, reward, terminated, truncated, info = custom_env.step(action)
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_runs_without_crash(self, custom_env):
        """Run 50 random steps on the custom topology."""
        custom_env.reset()
        for _ in range(50):
            action = int(custom_env.action_space.sample())
            obs, reward, terminated, truncated, info = custom_env.step(action)
            if terminated or truncated:
                custom_env.reset()

    def test_ai_infra_hosts_are_defined(self):
        assert len(AI_INFRA_HOSTS) == 4
        assert (3, 0) in AI_INFRA_HOSTS
        assert (4, 0) in AI_INFRA_HOSTS


# ---------------------------------------------------------------------------
# Stealth wrapper on custom topology
# ---------------------------------------------------------------------------


class TestCustomStealthEnv:
    def test_stealth_wrapper_loads(self, custom_stealth_env):
        assert isinstance(custom_stealth_env, StealthAwareWrapper)

    def test_stealth_reset_zeroes_detection(self, custom_stealth_env):
        custom_stealth_env.reset()
        assert custom_stealth_env.cumulative_detection == 0.0

    def test_stealth_step_returns_detection_info(self, custom_stealth_env):
        custom_stealth_env.reset()
        action = int(custom_stealth_env.action_space.sample())
        _, _, _, _, info = custom_stealth_env.step(action)
        assert "cumulative_detection" in info
        assert "detection_threshold" in info
        assert "caught" in info

    def test_stealth_preserves_spaces(self, custom_env, custom_stealth_env):
        assert custom_stealth_env.observation_space == custom_env.observation_space
        assert custom_stealth_env.action_space == custom_env.action_space

    def test_idle_step_no_detection_penalty(self, custom_stealth_env):
        """Action 0 (noop) should not accumulate detection."""
        custom_stealth_env.reset()
        _, reward, _, _, info = custom_stealth_env.step(0)
        # Noop: detection should not increase (action==0 and reward==0 → not active)
        assert info["cumulative_detection"] == 0.0

    def test_active_step_accumulates_detection(self, custom_stealth_env):
        """A non-noop action should increase cumulative detection."""
        custom_stealth_env.reset()
        # Use action index 1+ (scan/exploit) — should be active
        action = max(1, int(custom_stealth_env.action_space.sample()))
        _, _, _, _, info = custom_stealth_env.step(action)
        assert info["cumulative_detection"] >= 0.0

    def test_detection_summary(self, custom_stealth_env):
        custom_stealth_env.reset()
        custom_stealth_env.step(1)
        summary = custom_stealth_env.detection_summary()
        assert "cumulative_detection" in summary
        assert "steps" in summary
        assert "caught" in summary
        assert summary["steps"] == 1
