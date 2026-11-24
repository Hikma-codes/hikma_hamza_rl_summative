import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.agent_loader import load_all_agents
from environment.custom_env import SmartLearnEnv


def main():
    agents = load_all_agents()
    env = SmartLearnEnv()

    print("\n" + "=" * 60)
    print("TRAINED AGENT ACTION TEST")
    print("=" * 60)

    try:
        for name, agent in agents.items():
            observation, info = env.reset(
                seed=123,
                options={"profile": "beginner"},
            )

            action = agent.select_action(observation)

            if not env.action_space.contains(action):
                raise ValueError(
                    f"{name} selected invalid action: {action}"
                )

            _, reward, terminated, truncated, step_info = (
                env.step(action)
            )

            print(f"\nAgent: {name}")
            print(f"Selected action: {action}")
            print(f"Reward: {reward:.4f}")
            print(f"Success: {step_info['success']}")
            print(f"Mastery: {step_info['mean_mastery']:.4f}")
            print("Action valid: YES")

    finally:
        env.close()

    print("\nAll four agents passed the action test.")


if __name__ == "__main__":
    main()
