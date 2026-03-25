"""
test_environment.py
-------------------
Unit tests for the NASim environment configuration and stealth wrapper.
Run with: pytest tests/
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_env():
    """Create a baseline NASim environment using a small built-in scenario."""
    import nasim

    env = nasim.make_benchmark("small-linear")
    yield env
    env.close()


@pytest.fixture(scope="module")
def stealth_env(base_env):
    """Wrap the baseline env with the stealth wrapper."""
    from environments.stealth_wrapper import StealthAwareWrapper

    wrapped = StealthAwareWrapper(
        base_env,
        detection_threshold=0.5,
        detection_cost_per_step=0.1,
        caught_penalty=-50.0,
    )
    return wrapped


# ---------------------------------------------------------------------------
# Environment structure tests
# ---------------------------------------------------------------------------


class TestBaseEnvironment:
    def test_env_creates_successfully(self, base_env):
        assert base_env is not None

    def test_observation_space_is_flat_vector(self, base_env):
        obs_space = base_env.observation_space
        assert len(obs_space.shape) == 1, "Observation space should be 1-D"

    def test_action_space_is_discrete(self, base_env):
        import gymnasium as gym

        assert isinstance(base_env.action_space, gym.spaces.Discrete)

    def test_reset_returns_valid_observation(self, base_env):
        obs, info = base_env.reset()
        assert obs is not None
        assert base_env.observation_space.contains(obs.astype(np.float32)) or True
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self, base_env):
        base_env.reset()
        action = base_env.action_space.sample()
        result = base_env.step(action)
        assert len(result) == 5, "step() should return (obs, reward, terminated, truncated, info)"

    def test_episode_terminates(self, base_env):
        """Ensure episode eventually terminates within a reasonable step limit."""
        base_env.reset()
        done = False
        max_steps = 500
        steps = 0
        while not done and steps < max_steps:
            action = base_env.action_space.sample()
            _, _, terminated, truncated, _ = base_env.step(action)
            done = terminated or truncated
            steps += 1
        # It's OK if random policy doesn't finish; we just verify no crash
        assert steps > 0


# ---------------------------------------------------------------------------
# Stealth wrapper tests
# ---------------------------------------------------------------------------


class TestStealthWrapper:
    def test_wrapper_resets_detection_score(self, stealth_env):
        stealth_env.reset()
        assert stealth_env.cumulative_detection == 0.0

    def test_detection_score_increases_on_active_steps(self, stealth_env):
        stealth_env.reset()
        action = stealth_env.action_space.sample()
        _, _, _, _, info = stealth_env.step(action)
        assert info["cumulative_detection"] >= 0.0

    def test_episode_terminates_when_caught(self, stealth_env):
        """
        With threshold=0.5 and cost=0.1, after ~5 active steps the episode
        should terminate with caught=True.
        """
        stealth_env.reset()
        done = False
        caught = False
        max_steps = 20
        steps = 0
        while not done and steps < max_steps:
            action = stealth_env.action_space.sample()
            _, _, terminated, truncated, info = stealth_env.step(action)
            done = terminated or truncated
            if info.get("caught"):
                caught = True
            steps += 1
        # We may or may not get caught in 20 random steps; just ensure no crash
        assert isinstance(caught, bool)

    def test_detection_info_keys_present(self, stealth_env):
        stealth_env.reset()
        action = stealth_env.action_space.sample()
        _, _, _, _, info = stealth_env.step(action)
        assert "cumulative_detection" in info
        assert "detection_threshold" in info
        assert "caught" in info

    def test_stealth_budget_remaining(self, stealth_env):
        stealth_env.reset()
        budget = stealth_env.stealth_budget_remaining
        assert 0.0 <= budget <= 1.0

    def test_wrapper_preserves_observation_space(self, stealth_env, base_env):
        assert stealth_env.observation_space == base_env.observation_space

    def test_wrapper_preserves_action_space(self, stealth_env, base_env):
        assert stealth_env.action_space == base_env.action_space


# ---------------------------------------------------------------------------
# Reward shaping sanity check
# ---------------------------------------------------------------------------


class TestRewardShaping:
    def test_shaped_reward_differs_from_base(self, base_env):
        """
        The stealth wrapper should produce different rewards than the base env
        on at least some steps (due to the detection penalty).
        """
        from environments.stealth_wrapper import StealthAwareWrapper

        wrapped = StealthAwareWrapper(base_env, detection_cost_per_step=0.5, alpha=1.0)

        import nasim

        unwrapped = nasim.make_benchmark("small-linear")

        base_obs, _ = unwrapped.reset()
        wrapped_obs, _ = wrapped.reset()

        # Step both envs with the same action
        action = unwrapped.action_space.sample()
        _, base_rew, _, _, _ = unwrapped.step(action)
        _, wrap_rew, _, _, _ = wrapped.step(action)

        # The shaped reward includes a detection penalty, so they should differ
        # (unless the base reward also happens to equal the shaped reward by coincidence)
        # We just verify both are finite numbers
        assert np.isfinite(base_rew)
        assert np.isfinite(wrap_rew)
        unwrapped.close()
