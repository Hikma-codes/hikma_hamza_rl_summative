
import numpy as np

from gymnasium.utils.env_checker import check_env

from environment.custom_env import SmartLearnEnv


def test_gymnasium_api_compliance():
    """Check that the environment follows Gymnasium's API."""
    env = SmartLearnEnv()

    check_env(env, skip_render_check=True)

    env.close()


def test_seeded_runs_are_reproducible():
    """Identical seeds and actions should produce identical results."""
    env1 = SmartLearnEnv()
    env2 = SmartLearnEnv()

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)

    np.testing.assert_array_equal(obs1, obs2)

    actions = [0, 1, 2, 3, 4, 5, 6]

    for action in actions:
        result1 = env1.step(action)
        result2 = env2.step(action)

        np.testing.assert_array_equal(
            result1[0], result2[0]
        )

        assert result1[1] == result2[1]
        assert result1[2] == result2[2]
        assert result1[3] == result2[3]

        assert result1[4]["success"] == result2[4]["success"]

    env1.close()
    env2.close()


def test_observations_stay_within_valid_bounds():
    """Every observation must remain inside the declared space."""
    env = SmartLearnEnv(max_steps=20)

    observation, _ = env.reset(seed=42)

    assert env.observation_space.contains(observation)

    for action in [0, 1, 2, 3, 4, 5, 6]:
        observation, _, terminated, truncated, _ = env.step(action)

        assert env.observation_space.contains(observation)

        if terminated or truncated:
            break

    env.close()


def test_step_information_is_complete():
    """Check that each step reports useful evaluation information."""
    env = SmartLearnEnv()

    env.reset(seed=42)

    _, reward, _, _, info = env.step(1)

    required_fields = [
        "activity",
        "success",
        "success_probability",
        "target_skill",
        "skills",
        "mean_mastery",
        "confidence",
        "engagement",
        "fatigue",
        "steps",
        "termination_reason",
    ]

    for field in required_fields:
        assert field in info

    assert isinstance(reward, float)

    env.close()
