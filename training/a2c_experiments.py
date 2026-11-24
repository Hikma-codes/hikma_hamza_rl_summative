
import csv
import json
from pathlib import Path

import numpy as np

from training.a2c_training import DEFAULT_CONFIG, train_a2c


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


EXPERIMENTS = [
    {
        "name": "a2c_exp_01_baseline",
        "description": "Default A2C configuration",
        "config": {},
    },
    {
        "name": "a2c_exp_02_low_learning_rate",
        "description": "Lower learning rate",
        "config": {"learning_rate": 0.0003},
    },
    {
        "name": "a2c_exp_03_high_learning_rate",
        "description": "Higher learning rate",
        "config": {"learning_rate": 0.003},
    },
    {
        "name": "a2c_exp_04_lower_discount",
        "description": "Lower discount factor",
        "config": {"gamma": 0.90},
    },
    {
        "name": "a2c_exp_05_higher_discount",
        "description": "Higher discount factor",
        "config": {"gamma": 0.995},
    },
    {
        "name": "a2c_exp_06_lower_gae_lambda",
        "description": "Lower GAE lambda",
        "config": {"gae_lambda": 0.80},
    },
    {
        "name": "a2c_exp_07_higher_gae_lambda",
        "description": "Higher GAE lambda",
        "config": {"gae_lambda": 1.00},
    },
    {
        "name": "a2c_exp_08_lower_entropy",
        "description": "Lower entropy coefficient",
        "config": {"entropy_coeff": 0.001},
    },
    {
        "name": "a2c_exp_09_higher_entropy",
        "description": "Higher entropy coefficient",
        "config": {"entropy_coeff": 0.03},
    },
    {
        "name": "a2c_exp_10_smaller_network",
        "description": "Smaller actor and critic networks",
        "config": {"hidden_size": 64},
    },
]


def run_experiments():

    summaries = []
    detailed_results = []

    print("\n" + "=" * 65)
    print("SMARTLEARN GHANA - A2C HYPERPARAMETER EXPERIMENTS")
    print("=" * 65)

    print(f"Number of experiments: {len(EXPERIMENTS)}")
    print(f"Episodes per experiment: {DEFAULT_CONFIG['episodes']}")
    print(
        f"Total training episodes: "
        f"{len(EXPERIMENTS) * DEFAULT_CONFIG['episodes']}"
    )

    for index, experiment in enumerate(EXPERIMENTS, start=1):

        experiment_name = experiment["name"]

        config = {
            **DEFAULT_CONFIG,
            **experiment["config"],
        }

        print("\n" + "#" * 65)
        print(f"EXPERIMENT {index}/{len(EXPERIMENTS)}")
        print(experiment_name)
        print(experiment["description"])
        print("#" * 65)

        history = train_a2c(
            config=config,
            run_name=experiment_name,
        )

        final_ten = history[-10:]

        summary = {
            "experiment": experiment_name,
            "description": experiment["description"],
            "average_reward": float(
                np.mean([
                    item["episode_reward"]
                    for item in history
                ])
            ),
            "final_10_average_reward": float(
                np.mean([
                    item["episode_reward"]
                    for item in final_ten
                ])
            ),
            "final_10_average_mastery": float(
                np.mean([
                    item["final_mastery"]
                    for item in final_ten
                ])
            ),
            "final_10_average_success_rate": float(
                np.mean([
                    item["success_rate"]
                    for item in final_ten
                ])
            ),
            "average_actor_loss": float(
                np.mean([
                    item["actor_loss"]
                    for item in history
                ])
            ),
            "average_critic_loss": float(
                np.mean([
                    item["critic_loss"]
                    for item in history
                ])
            ),
            "average_entropy_loss": float(
                np.mean([
                    item["entropy_loss"]
                    for item in history
                ])
            ),
            "learning_rate": config["learning_rate"],
            "gamma": config["gamma"],
            "gae_lambda": config["gae_lambda"],
            "entropy_coeff": config["entropy_coeff"],
            "hidden_size": config["hidden_size"],
            "seed": config["seed"],
        }

        summaries.append(summary)

        detailed_results.append({
            "experiment": experiment_name,
            "description": experiment["description"],
            "configuration": config,
            "training_history": history,
        })

        print("\nExperiment summary:")
        print(
            f"Average reward: "
            f"{summary['average_reward']:.4f}"
        )
        print(
            f"Final 10 reward: "
            f"{summary['final_10_average_reward']:.4f}"
        )
        print(
            f"Final 10 mastery: "
            f"{summary['final_10_average_mastery']:.4f}"
        )

    summary_path = (
        RESULTS_DIR / "a2c_experiment_summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summaries[0].keys(),
        )

        writer.writeheader()
        writer.writerows(summaries)

    details_path = (
        RESULTS_DIR / "a2c_experiment_details.json"
    )

    with open(
        details_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            detailed_results,
            file,
            indent=4,
        )

    print("\n" + "=" * 65)
    print("ALL A2C EXPERIMENTS COMPLETED")
    print("=" * 65)

    print(f"Summary saved to: {summary_path}")
    print(f"Detailed results saved to: {details_path}")

    print("\nFINAL A2C EXPERIMENT COMPARISON")

    for result in sorted(
        summaries,
        key=lambda item: item["final_10_average_reward"],
        reverse=True,
    ):

        print(
            f"{result['experiment']:<38} "
            f"Reward: {result['final_10_average_reward']:>8.4f} "
            f"Mastery: {result['final_10_average_mastery']:.4f} "
            f"Success: {result['final_10_average_success_rate']:.2%}"
        )


if __name__ == "__main__":
    run_experiments()
