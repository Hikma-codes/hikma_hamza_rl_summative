
from collections.abc import Callable
from typing import Any

import numpy as np

from environment.custom_env import SmartLearnEnv
from evaluation.metrics import (
    aggregate_episode_metrics,
    summarize_episode,
)


Policy = Callable[[np.ndarray], int]

DEFAULT_PROFILES = (
    "beginner",
    "developing",
    "intermediate",
    "uneven",
)


def evaluate_agent(
    policy: Policy,
    agent_name: str,
    profiles: tuple[str, ...] = DEFAULT_PROFILES,
    seeds: tuple[int, ...] = (101, 202, 303, 404, 505),
    max_steps: int = 20,
) -> dict[str, Any]:
    """
    Evaluate one trained policy on a shared set of learner profiles and seeds.

    The policy receives an observation and must return one discrete action.
    The same profile/seed combinations should be used for every algorithm.
    """

    if not profiles:
        raise ValueError("At least one learner profile must be provided.")

    if not seeds:
        raise ValueError("At least one evaluation seed must be provided.")

    env = SmartLearnEnv(max_steps=max_steps)
    episode_results = []

    try:
        for profile in profiles:
            for seed in seeds:
                observation, initial_info = env.reset(
                    seed=seed,
                    options={"profile": profile},
                )

                rewards = []
                step_infos = []
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    action = policy(observation)
                    action = int(action)

                    if not env.action_space.contains(action):
                        raise ValueError(
                            f"{agent_name} selected invalid action: {action}"
                        )

                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(action)

                    rewards.append(float(reward))
                    step_infos.append(info)

                episode_result = summarize_episode(
                    profile=profile,
                    initial_info=initial_info,
                    step_infos=step_infos,
                    rewards=rewards,
                    terminated=terminated,
                    truncated=truncated,
                )

                episode_results.append(episode_result)

    finally:
        env.close()

    aggregate_results = aggregate_episode_metrics(episode_results)

    return {
        "agent_name": agent_name,
        "episode_results": episode_results,
        "aggregate_results": aggregate_results,
    }


def print_evaluation_report(results: dict[str, Any]) -> None:
    """Print a readable summary of an agent's evaluation."""

    summary = results["aggregate_results"]

    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT: {results['agent_name']}")
    print("=" * 60)

    print(f"Episodes evaluated: {summary['episodes_evaluated']}")
    print(
        f"Average episode reward: "
        f"{summary['mean_episode_return']:.4f}"
    )
    print(
        f"Average mastery improvement: "
        f"{summary['mean_mean_mastery_gain']:.4f}"
    )
    print(
        f"Average success rate: "
        f"{summary['mean_success_rate']:.2%}"
    )
    print(
        f"Average confidence: "
        f"{summary['mean_average_confidence']:.4f}"
    )
    print(
        f"Average engagement: "
        f"{summary['mean_average_engagement']:.4f}"
    )
    print(
        f"Average fatigue: "
        f"{summary['mean_average_fatigue']:.4f}"
    )

    print("\nAverage improvement by skill:")

    for skill in (
        "vocabulary",
        "listening",
        "comprehension",
        "sentence_building",
    ):
        print(
            f"  {skill}: "
            f"{summary[f'mean_{skill}_gain']:.4f}"
        )

    print("\nEpisode ending reasons:")

    for reason, count in summary["termination_counts"].items():
        print(f"  {reason}: {count}")

    print("=" * 60)
