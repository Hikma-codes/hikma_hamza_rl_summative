import json
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("evaluation/results/independent_evaluation.json")
OUTPUT_DIR = Path("evaluation/results/analysis")

PROFILES = ["beginner", "developing", "intermediate", "uneven"]
SEEDS = list(range(101, 111))


def main():
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        results = json.load(file)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    episode_rows = []
    activity_rows = []

    for agent_name, result in results.items():
        episodes = result["episode_results"]

        if len(episodes) != len(PROFILES) * len(SEEDS):
            raise ValueError(
                f"{agent_name} has {len(episodes)} episodes; "
                "expected 40."
            )

        # Evaluation was run profile-by-profile, with ten seeds
        # for each profile. The saved JSON does not store seed IDs.
        for index, episode in enumerate(episodes):
            profile = PROFILES[index // len(SEEDS)]
            seed = SEEDS[index % len(SEEDS)]

            row = {
                "agent": agent_name,
                "profile": profile,
                "seed": seed,
                "reward": episode["episode_return"],
                "mastery_gain": episode["mean_mastery_gain"],
                "success_rate": episode["success_rate"],
                "confidence": episode["average_confidence"],
                "engagement": episode["average_engagement"],
                "fatigue": episode["average_fatigue"],
                "episode_length": episode["episode_length"],
            }

            for skill, gain in episode["skill_gains"].items():
                row[f"{skill}_gain"] = gain

            episode_rows.append(row)

            for activity, count in episode["activity_counts"].items():
                activity_rows.append({
                    "agent": agent_name,
                    "profile": profile,
                    "activity": activity,
                    "count": count,
                })

    episodes_df = pd.DataFrame(episode_rows)
    activities_df = pd.DataFrame(activity_rows)

    # 1. Performance by learner profile
    profile_summary = (
        episodes_df.groupby(["agent", "profile"])
        .agg(
            average_reward=("reward", "mean"),
            reward_std=("reward", "std"),
            mastery_gain=("mastery_gain", "mean"),
            success_rate=("success_rate", "mean"),
            confidence=("confidence", "mean"),
            engagement=("engagement", "mean"),
            fatigue=("fatigue", "mean"),
        )
        .round(4)
        .reset_index()
    )

    # 2. Most frequently selected activities
    activity_summary = (
        activities_df.groupby(["agent", "activity"])["count"]
        .sum()
        .reset_index()
        .sort_values(["agent", "count"], ascending=[True, False])
    )

    # 3. Consistency across seeds.
    # Each seed contains one episode from each of the four profiles.
    seed_summary = (
        episodes_df.groupby(["agent", "seed"])
        .agg(
            average_reward=("reward", "mean"),
            average_mastery_gain=("mastery_gain", "mean"),
            average_success_rate=("success_rate", "mean"),
        )
        .reset_index()
    )

    consistency_summary = (
        seed_summary.groupby("agent")
        .agg(
            mean_seed_reward=("average_reward", "mean"),
            std_seed_reward=("average_reward", "std"),
            lowest_seed_reward=("average_reward", "min"),
            highest_seed_reward=("average_reward", "max"),
            mean_seed_mastery_gain=("average_mastery_gain", "mean"),
            std_seed_mastery_gain=("average_mastery_gain", "std"),
        )
        .round(4)
        .reset_index()
    )

    # 4. Overall skill improvement
    skill_columns = [
        column for column in episodes_df.columns
        if column.endswith("_gain") and column != "mastery_gain"
    ]

    skill_summary = (
        episodes_df.groupby("agent")[skill_columns]
        .mean()
        .round(4)
        .reset_index()
    )

    # Save all analysis tables.
    episodes_df.to_csv(OUTPUT_DIR / "episode_level_results.csv", index=False)
    profile_summary.to_csv(OUTPUT_DIR / "profile_comparison.csv", index=False)
    activity_summary.to_csv(OUTPUT_DIR / "activity_selection.csv", index=False)
    seed_summary.to_csv(OUTPUT_DIR / "seed_level_results.csv", index=False)
    consistency_summary.to_csv(
        OUTPUT_DIR / "seed_consistency.csv", index=False)
    skill_summary.to_csv(OUTPUT_DIR / "skill_improvements.csv", index=False)

    print("\n" + "=" * 75)
    print("1. PERFORMANCE BY LEARNER PROFILE")
    print("=" * 75)
    print(profile_summary.to_string(index=False))

    print("\n" + "=" * 75)
    print("2. MOST FREQUENTLY SELECTED ACTIVITIES")
    print("=" * 75)
    print(activity_summary.to_string(index=False))

    print("\n" + "=" * 75)
    print("3. CONSISTENCY ACROSS THE TEN SEEDS")
    print("=" * 75)
    print(consistency_summary.to_string(index=False))

    print("\n" + "=" * 75)
    print("4. IMPROVEMENT BY SKILL")
    print("=" * 75)
    print(skill_summary.to_string(index=False))

    print("\nAnalysis files saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
