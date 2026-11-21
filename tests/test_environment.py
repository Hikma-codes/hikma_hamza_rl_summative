
import numpy as np

from environment.custom_env import SmartLearnEnv


def test_environment_reset():
    env = SmartLearnEnv()
    observation, info = env.reset(seed=42)

    assert observation.shape == (11,)
    assert env.observation_space.contains(observation)
    assert info["profile"] == "beginner"


def test_action_space():
    env = SmartLearnEnv()

    assert env.action_space.n == 7

    for action in range(7):
        assert action in env.action_space


def test_environment_step():
    env = SmartLearnEnv()
    observation, info = env.reset(seed=42)

    next_observation, reward, terminated, truncated, info = (
        env.step(0)
    )

    assert next_observation.shape == (11,)
    assert env.observation_space.contains(next_observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["steps"] == 1


def test_state_changes_after_activity():
    env = SmartLearnEnv()
    env.reset(seed=42)

    initial_fatigue = env.state["fatigue"]

    env.step(0)

    assert env.state["fatigue"] > initial_fatigue


def test_all_learner_profiles():
    env = SmartLearnEnv()

    for profile in env.PROFILES:
        observation, info = env.reset(
            seed=42,
            options={"profile": profile},
        )

        assert observation.shape == (11,)
        assert info["profile"] == profile


def test_episode_has_a_time_limit():
    env = SmartLearnEnv(max_steps=3)
    env.reset(seed=42)

    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(0)

    assert terminated or truncated
