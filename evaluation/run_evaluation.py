
import json
from pathlib import Path

import numpy as np

from evaluation.agent_loader import load_all_agents
from evaluation.evaluate_agents import (
    evaluate_agent,
    print_evaluation_report,
)


PROFILES = (
    "beginner",
    "developing",
    "intermediate",
    "uneven",
)

SEEDS = tuple(range(101, 111))
MAX_STEPS = 20


def make_random_policy(seed=2026):
    rng = np.random.default_rng(seed)

    def select_action(observation):
        return int(rng.integers(0, 7))

    return select_action


def main():
    print("=" * 65)
    print("SMARTLEARN GHANA: INDEPENDENT AGENT EVALUATION")
    print("=" * 65)
    print(f"Learner profiles: {len(PROFILES)}")
    print(f"Random seeds: {len(SEEDS)}")
    print(f"Episodes per agent: {len(PROFILES) * len(SEEDS)}")
    print(f"Maximum steps per episode: {MAX_STEPS}")

    agents = load_all_agents()
    all_results = {}

    for name, agent in agents.items():
        print(f"\n\nEvaluating {name}...")
        result = evaluate_agent(
            policy=agent.select_action,
            agent_name=name,
            profiles=PROFILES,
            seeds=SEEDS,
            max_steps=MAX_STEPS,
        )
        all_results[name] = result
        print_evaluation_report(result)

    print("\n\nEvaluating RANDOM baseline...")
    random_result = evaluate_agent(
        policy=make_random_policy(),
        agent_name="Random",
        profiles=PROFILES,
        seeds=SEEDS,
        max_steps=MAX_STEPS,
    )
    all_results["Random"] = random_result
    print_evaluation_report(random_result)

    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "independent_evaluation.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            indent=4,
            default=lambda value: (
                value.item()
                if isinstance(value, np.generic)
                else str(value)
            ),
        )

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {output_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
