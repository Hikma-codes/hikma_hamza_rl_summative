
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from training.dqn_training import DQNNetwork
from training.reinforce_training import build_actor as build_reinforce_actor
from training.ppo_training import build_actor as build_ppo_actor
from training.a2c_training import build_actor as build_a2c_actor

from environment.custom_env import SmartLearnEnv


CHECKPOINTS = {
    "DQN": Path("models/dqn/dqn_exp_02_low_learning_rate.pt"),
    "REINFORCE": Path("models/reinforce/reinforce_exp_07_larger_network.pt"),
    "PPO": Path("models/ppo/ppo_exp_05_lower_gae_lambda.pt"),
    "A2C": Path("models/a2c/a2c_exp_05_higher_discount.pt"),
}


class TrainedAgent:
    """Loads a trained checkpoint and selects educational activities."""

    def __init__(self, algorithm, checkpoint_path):
        self.algorithm = algorithm
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device("cpu")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(self.checkpoint_path)

        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        config = self.checkpoint["config"]
        hidden_size = config["hidden_size"]

        if algorithm == "DQN":
            self.model = DQNNetwork(
                observation_size=11,
                action_size=7,
                hidden_size=hidden_size,
            )

            saved_weights = self.checkpoint["model_state_dict"]

            corrected_weights = {
                key.replace("module.0.module.", "", 1): value
                for key, value in saved_weights.items()
            }

            self.model.load_state_dict(corrected_weights)

        else:
            env = SmartLearnEnv()

            try:
                action_spec = env.action_space
                from torchrl.envs import GymWrapper

                wrapped_env = GymWrapper(env)
                action_spec = wrapped_env.action_spec

                if algorithm == "REINFORCE":
                    self.model = build_reinforce_actor(
                        action_spec,
                        hidden_size,
                    )

                elif algorithm == "PPO":
                    self.model = build_ppo_actor(
                        action_spec,
                        hidden_size,
                    )

                elif algorithm == "A2C":
                    self.model = build_a2c_actor(
                        action_spec,
                        hidden_size,
                    )

                else:
                    raise ValueError(
                        f"Unknown algorithm: {algorithm}"
                    )

                self.model.load_state_dict(
                    self.checkpoint["actor_state_dict"]
                )

            finally:
                env.close()

        self.model.to(self.device)
        self.model.eval()

    def select_action(self, observation):
        observation_tensor = torch.as_tensor(
            np.asarray(observation, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            if self.algorithm == "DQN":
                action_values = self.model(observation_tensor)
                action = int(action_values.argmax(dim=-1).item())

            else:
                # Run the underlying neural network directly.
                actor_network = self.model.module[0].module
                logits = actor_network(observation_tensor)
                action = int(logits.argmax(dim=-1).item())

        return action


def load_all_agents():
    """Load all four selected trained agents."""

    agents = {}

    for algorithm, checkpoint_path in CHECKPOINTS.items():
        print(f"Loading {algorithm}...")

        agents[algorithm] = TrainedAgent(
            algorithm,
            checkpoint_path,
        )

        print(f"Successfully loaded {algorithm}.")

    return agents
