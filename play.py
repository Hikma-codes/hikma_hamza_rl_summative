
import argparse

import pygame
import numpy as np

from environment.custom_env import SmartLearnEnv
from environment.rendering import SmartLearnRenderer
from evaluation.agent_loader import TrainedAgent, CHECKPOINTS


def main():
    parser = argparse.ArgumentParser(
        description="Run the SmartLearn Ghana learning simulation."
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="DQN",
        choices=["DQN", "REINFORCE", "PPO", "A2C", "RANDOM"],
        help="Choose the trained agent or random baseline.",
    )

    args = parser.parse_args()
    selected_agent = args.agent

    # Load the selected trained agent.
    agent = None

    if selected_agent != "RANDOM":
        print(f"\nLoading trained {selected_agent} agent...")

        agent = TrainedAgent(
            selected_agent,
            CHECKPOINTS[selected_agent],
        )

        print(f"{selected_agent} loaded successfully.\n")

    else:
        print("\nUsing the random baseline policy.\n")

    env = SmartLearnEnv(
        max_steps=20,
        render_mode=None,
    )

    renderer = SmartLearnRenderer()
    rng = np.random.default_rng(42)

    episode = 1
    episode_reward = 0.0

    observation, reset_info = env.reset(
        seed=42,
        options={"profile": "beginner"},
    )

    last_info = None
    paused = False
    completed_at = None

    last_step_time = pygame.time.get_ticks()

    pygame.display.set_caption(
        f"SmartLearn Ghana | Agent: {selected_agent}"
    )

    print("SmartLearn Ghana simulation started.")
    print(f"Selected policy: {selected_agent}")
    print("Press SPACE to pause or resume.")
    print("Press R to restart the episode.")
    print("Press ESC to exit.\n")

    running = True

    try:
        while running:

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_SPACE:
                        paused = not paused

                    elif event.key == pygame.K_r:
                        observation, reset_info = env.reset(
                            seed=int(rng.integers(0, 100000)),
                            options={"profile": "beginner"},
                        )

                        episode_reward = 0.0
                        last_info = None
                        completed_at = None
                        last_step_time = pygame.time.get_ticks()

                        print(f"\nEpisode {episode} restarted.")

            current_time = pygame.time.get_ticks()

            # Run one learning activity every 1.2 seconds.
            if (
                running
                and not paused
                and completed_at is None
                and current_time - last_step_time >= 1200
            ):

                # Select an action using the chosen policy.
                if selected_agent == "RANDOM":
                    action = int(rng.integers(0, 7))
                else:
                    action = agent.select_action(observation)

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    last_info,
                ) = env.step(action)

                episode_reward += reward

                print(
                    f"Agent: {selected_agent} | "
                    f"Episode: {episode} | "
                    f"Step: {env.steps} | "
                    f"Action: {last_info['activity']} | "
                    f"Success: {last_info['success']} | "
                    f"Reward: {reward:.3f} | "
                    f"Mastery: {last_info['mean_mastery']:.3f}"
                )

                last_step_time = current_time

                if terminated or truncated:
                    completed_at = current_time

                    print(
                        f"\nEpisode {episode} finished: "
                        f"{last_info['termination_reason']}"
                    )

                    print(
                        f"Total episode reward: {episode_reward:.3f}\n"
                    )

            # Start the next episode after displaying the result.
            if (
                completed_at is not None
                and current_time - completed_at >= 2200
            ):

                episode += 1

                observation, reset_info = env.reset(
                    seed=int(rng.integers(0, 100000)),
                    options={"profile": "beginner"},
                )

                episode_reward = 0.0
                last_info = None
                completed_at = None
                last_step_time = current_time

            renderer.draw(
                env=env,
                info=last_info,
                episode=episode,
                episode_reward=episode_reward,
                paused=paused,
                baseline=(selected_agent == "RANDOM"),
            )

    finally:
        renderer.close()
        env.close()

    print("SmartLearn Ghana simulation closed.")


if __name__ == "__main__":
    main()
