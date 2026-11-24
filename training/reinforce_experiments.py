
"""
Run 10 REINFORCE hyperparameter experiments for SmartLearn Ghana.

Results:
- experiments/results/reinforce_experiment_summary.csv
- experiments/results/reinforce_experiment_details.json
"""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.reinforce_training import (
    DEFAULT_CONFIG,
    train_reinforce,
)


# 1. EXPERIMENT CONFIGURATIONS


EXPERIMENTS = [
    {
        "name": "reinforce_exp_01_baseline",
        "description": "Standard REINFORCE configuration",
        "changes": {},
    },
    {
        "name": "reinforce_exp_02_low_learning_rate",
        "description": "Lower learning rate",
        "changes": {"learning_rate": 0.0001},
    },
    {
        "name": "reinforce_exp_03_high_learning_rate",
        "description": "Higher learning rate",
        "changes": {"learning_rate": 0.003},
    },
    {
        "name": "reinforce_exp_04_lower_discount",
        "description": "Lower discount factor",
        "changes": {"gamma": 0.90},
    },
    {
        "name": "reinforce_exp_05_higher_discount",
        "description": "Higher discount factor",
        "changes": {"gamma": 0.995},
    },
    {
        "name": "reinforce_exp_06_smaller_network",
        "description": "Smaller hidden network",
        "changes": {"hidden_size": 64},
    },
    {
        "name": "reinforce_exp_07_larger_network",
        "description": "Larger hidden network",
        "changes": {"hidden_size": 256},
    },
    {
        "name": "reinforce_exp_08_stronger_gradient_clipping",
        "description": "Stronger gradient clipping",
        "changes": {"max_grad_norm": 0.5},
    },
    {
        "name": "reinforce_exp_09_weaker_gradient_clipping",
        "description": "Weaker gradient clipping",
        "changes": {"max_grad_norm": 2.0},
    },
    {
        "name": "reinforce_exp_10_lower_gae_lambda",
        "description": "Lower GAE lambda",
        "changes": {"gae_lambda": 0.90},
    },
]


# 2. OUTPUT DIRECTORY


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# 3. RUN EXPERIMENTS


def run_reinforce_experiments():

    all_results = []

    print("\n" + "=" * 70)
    print("SMARTLEARN GHANA - REINFORCE HYPERPARAMETER EXPERIMENTS")
    print("=" * 70)

    print(f"Total experiments: {len(EXPERIMENTS)}")
    print("Episodes per experiment: 100")
    print("Total training episodes: 1,000")

    for index, experiment in enumerate(EXPERIMENTS, start=1):

        print("\n" + "=" * 70)
        print(
            f"EXPERIMENT {index}/{len(EXPERIMENTS)}: "
            f"{experiment['name']}"
        )
        print(experiment["description"])
        print("=" * 70)

        config = copy.deepcopy(DEFAULT_CONFIG)

        config.update(experiment["changes"])

        training_history = train_reinforce(
            config=config,
            run_name=experiment["name"],
        )

        final_10 = training_history[-10:]

        result = {
            "experiment": experiment["name"],
            "description": experiment["description"],
            "average_reward": float(
                np.mean([
                    item["episode_reward"]
                    for item in training_history
                ])
            ),
            "final_10_average_reward": float(
                np.mean([
                    item["episode_reward"]
                    for item in final_10
                ])
            ),
            "average_final_mastery": float(
                np.mean([
                    item["final_mastery"]
                    for item in training_history
                ])
            ),
            "final_10_average_mastery": float(
                np.mean([
                    item["final_mastery"]
                    for item in final_10
                ])
            ),
            "average_success_rate": float(
                np.mean([
                    item["success_rate"]
                    for item in training_history
                ])
            ),
            "final_10_average_success_rate": float(
                np.mean([
                    item["success_rate"]
                    for item in final_10
                ])
            ),
            "average_actor_loss": float(
                np.mean([
                    item["actor_loss"]
                    for item in training_history
                ])
            ),
            "average_critic_loss": float(
                np.mean([
                    item["critic_loss"]
                    for item in training_history
                ])
            ),
            "final_actor_loss": float(
                training_history[-1]["actor_loss"]
            ),
            "final_critic_loss": float(
                training_history[-1]["critic_loss"]
            ),
            "learning_rate": config["learning_rate"],
            "gamma": config["gamma"],
            "hidden_size": config["hidden_size"],
            "max_grad_norm": config["max_grad_norm"],
            "seed": config["seed"],
        }

        all_results.append(result)

        print("\nEXPERIMENT SUMMARY")
        print(f"Average reward: {result['average_reward']:.4f}")
        print(
            "Final 10 episode reward: "
            f"{result['final_10_average_reward']:.4f}"
        )
        print(
            "Final 10 mastery: "
            f"{result['final_10_average_mastery']:.4f}"
        )
        print(
            "Final 10 success rate: "
            f"{result['final_10_average_success_rate']:.2%}"
        )

    # 4. SAVE RESULTS

    summary_df = pd.DataFrame(all_results)

    csv_path = RESULTS_DIR / "reinforce_experiment_summary.csv"

    json_path = RESULTS_DIR / "reinforce_experiment_details.json"

    summary_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=4)

    # 5. DISPLAY FINAL COMPARISON

    print("\n" + "=" * 100)
    print("REINFORCE EXPERIMENT COMPARISON")
    print("=" * 100)

    print(
        summary_df[
            [
                "experiment",
                "average_reward",
                "final_10_average_reward",
                "final_10_average_mastery",
                "final_10_average_success_rate",
                "average_actor_loss",
                "average_critic_loss",
            ]
        ].to_string(index=False)
    )

    best_experiment = summary_df.loc[
        summary_df["final_10_average_reward"].idxmax()
    ]

    print("\nBEST EXPERIMENT BY FINAL 10-EPISODE REWARD")
    print(f"Experiment: {best_experiment['experiment']}")
    print(
        "Final 10 average reward: "
        f"{best_experiment['final_10_average_reward']:.4f}"
    )

    print(f"\nCSV results saved to: {csv_path}")
    print(f"JSON results saved to: {json_path}")

    return summary_df


# 6. RUN EXPERIMENTS

if __name__ == "__main__":
    run_reinforce_experiments()
