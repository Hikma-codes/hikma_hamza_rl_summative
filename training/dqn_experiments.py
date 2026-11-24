
import json
from pathlib import Path

import pandas as pd

from training.dqn_training import train_dqn, DEFAULT_CONFIG


# Each experiment changes one or more meaningful DQN hyperparameters.
# Experiment 1 reproduces the default baseline configuration.
DQN_EXPERIMENTS = [
    {
        "name": "dqn_exp_01_baseline",
        "description": "Default DQN configuration",
        "config": {},
    },
    {
        "name": "dqn_exp_02_low_learning_rate",
        "description": "Lower learning rate",
        "config": {"learning_rate": 0.0001},
    },
    {
        "name": "dqn_exp_03_high_learning_rate",
        "description": "Higher learning rate",
        "config": {"learning_rate": 0.003},
    },
    {
        "name": "dqn_exp_04_lower_discount",
        "description": "Lower discount factor",
        "config": {"gamma": 0.90},
    },
    {
        "name": "dqn_exp_05_larger_batch",
        "description": "Larger replay batch size",
        "config": {"batch_size": 64},
    },
    {
        "name": "dqn_exp_06_slower_exploration_decay",
        "description": "Agent explores for longer",
        "config": {"epsilon_decay": 0.998},
    },
    {
        "name": "dqn_exp_07_faster_exploration_decay",
        "description": "Agent reduces exploration faster",
        "config": {"epsilon_decay": 0.98},
    },
    {
        "name": "dqn_exp_08_faster_target_update",
        "description": "Faster target network updates",
        "config": {"target_update_tau": 0.02},
    },
    {
        "name": "dqn_exp_09_smaller_network",
        "description": "Smaller neural network",
        "config": {"hidden_size": 64},
    },
    {
        "name": "dqn_exp_10_larger_network",
        "description": "Larger neural network",
        "config": {"hidden_size": 256},
    },
]


def run_dqn_experiments(episodes=None):
    """Run all DQN configurations and save their results."""

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    detailed_results = []

    print("\n" + "=" * 65)
    print("SMARTLEARN GHANA — DQN HYPERPARAMETER EXPERIMENTS")
    print("=" * 65)

    for index, experiment in enumerate(DQN_EXPERIMENTS, start=1):
        name = experiment["name"]

        config = {
            **DEFAULT_CONFIG,
            **experiment["config"],
        }

        if episodes is not None:
            config["episodes"] = episodes

        print(f"\nExperiment {index}/{len(DQN_EXPERIMENTS)}")
        print(f"Name: {name}")
        print(f"Description: {experiment['description']}")
        print(f"Configuration: {config}")
        print("-" * 65)

        result = train_dqn(
            config=config,
            run_name=name,
        )

        history = result["history"]

        rewards = [
            item["episode_reward"]
            for item in history
        ]

        losses = [
            item["mean_loss"]
            for item in history
            if item["mean_loss"] is not None
        ]

        final_rewards = rewards[-10:] if rewards else []

        summary = {
            "experiment": name,
            "description": experiment["description"],
            "episodes": len(history),
            "learning_rate": config["learning_rate"],
            "gamma": config["gamma"],
            "batch_size": config["batch_size"],
            "buffer_size": config["buffer_size"],
            "epsilon_decay": config["epsilon_decay"],
            "target_update_tau": config["target_update_tau"],
            "hidden_size": config["hidden_size"],
            "average_reward": round(
                sum(rewards) / len(rewards), 4
            ) if rewards else None,
            "final_10_average_reward": round(
                sum(final_rewards) / len(final_rewards), 4
            ) if final_rewards else None,
            "average_loss": round(
                sum(losses) / len(losses), 6
            ) if losses else None,
            "final_epsilon": history[-1]["epsilon"] if history else None,
            "checkpoint": result["checkpoint"],
        }

        summary_rows.append(summary)

        detailed_results.append({
            "experiment": name,
            "description": experiment["description"],
            "configuration": config,
            "summary": summary,
            "history": history,
        })

        print("\nExperiment result:")
        print(f"Average reward: {summary['average_reward']}")
        print(
            "Final 10-episode average reward:",
            summary["final_10_average_reward"],
        )
        print(f"Average loss: {summary['average_loss']}")

    # Save a spreadsheet-friendly summary.
    summary_path = results_dir / "dqn_experiment_summary.csv"
    pd.DataFrame(summary_rows).to_csv(
        summary_path,
        index=False,
    )

    # Save complete episode-by-episode results.
    details_path = results_dir / "dqn_experiment_details.json"
    with open(details_path, "w", encoding="utf-8") as file:
        json.dump(
            detailed_results,
            file,
            indent=4,
        )

    print("\n" + "=" * 65)
    print("ALL DQN EXPERIMENTS COMPLETED")
    print("=" * 65)
    print(f"Summary saved to: {summary_path}")
    print(f"Detailed results saved to: {details_path}")

    results_table = pd.DataFrame(summary_rows)
    print("\nDQN EXPERIMENT COMPARISON")
    print(
        results_table[
            [
                "experiment",
                "average_reward",
                "final_10_average_reward",
                "average_loss",
                "final_epsilon",
            ]
        ].to_string(index=False)
    )

    return results_table


if __name__ == "__main__":
    run_dqn_experiments()
