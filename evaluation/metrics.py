from collections import Counter
from statistics import mean, pstdev
from typing import Any


SKILL_NAMES = (
    "vocabulary",
    "listening",
    "comprehension",
    "sentence_building",
)


def summarize_episode(
    profile: str,
    initial_info: dict[str, Any],
    step_infos: list[dict[str, Any]],
    rewards: list[float],
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    """
    Calculate performance metrics for one completed learning episode.

    All learner measurements come from SmartLearnEnv's info dictionaries.
    """

    if not step_infos:
        raise ValueError(
            "An episode must contain at least one environment step.")

    if len(step_infos) != len(rewards):
        raise ValueError(
            "The number of step information records must match the rewards."
        )

    initial_skills = initial_info["skills"]
    final_info = step_infos[-1]
    final_skills = final_info["skills"]

    skill_gains = {
        skill: round(
            float(final_skills[skill]) - float(initial_skills[skill]), 4
        )
        for skill in SKILL_NAMES
    }

    success_count = sum(
        bool(info["success"]) for info in step_infos
    )

    activity_counts = dict(
        Counter(info["activity"] for info in step_infos)
    )

    episode_return = sum(float(reward) for reward in rewards)

    initial_mastery = mean(
        float(initial_skills[skill]) for skill in SKILL_NAMES
    )

    final_mastery = mean(
        float(final_skills[skill]) for skill in SKILL_NAMES
    )

    return {
        "profile": profile,
        "episode_return": round(episode_return, 4),
        "episode_length": len(rewards),
        "initial_mean_mastery": round(initial_mastery, 4),
        "final_mean_mastery": round(final_mastery, 4),
        "mean_mastery_gain": round(
            final_mastery - initial_mastery, 4
        ),
        "skill_gains": skill_gains,
        "success_rate": round(success_count / len(step_infos), 4),
        "average_confidence": round(
            mean(float(info["confidence"]) for info in step_infos), 4
        ),
        "average_engagement": round(
            mean(float(info["engagement"]) for info in step_infos), 4
        ),
        "average_fatigue": round(
            mean(float(info["fatigue"]) for info in step_infos), 4
        ),
        "activity_counts": activity_counts,
        "termination_reason": final_info["termination_reason"],
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def aggregate_episode_metrics(
    episode_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate results across multiple evaluation episodes.

    Returns means and population standard deviations for key metrics.
    """

    if not episode_results:
        raise ValueError("At least one episode result is required.")

    numeric_metrics = (
        "episode_return",
        "episode_length",
        "initial_mean_mastery",
        "final_mean_mastery",
        "mean_mastery_gain",
        "success_rate",
        "average_confidence",
        "average_engagement",
        "average_fatigue",
    )

    summary = {
        "episodes_evaluated": len(episode_results),
    }

    for metric in numeric_metrics:
        values = [
            float(result[metric])
            for result in episode_results
        ]

        summary[f"mean_{metric}"] = round(mean(values), 4)
        summary[f"std_{metric}"] = round(pstdev(values), 4)

    for skill in SKILL_NAMES:
        gains = [
            float(result["skill_gains"][skill])
            for result in episode_results
        ]

        summary[f"mean_{skill}_gain"] = round(mean(gains), 4)

    termination_counts = Counter(
        result["termination_reason"]
        for result in episode_results
    )

    summary["termination_counts"] = dict(termination_counts)

    return summary
