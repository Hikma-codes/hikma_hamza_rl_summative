
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tensordict import TensorDict

from torchrl.envs import GymWrapper
from torchrl.modules import (
    ProbabilisticActor,
    OneHotCategorical,
    ValueOperator,
)
from torchrl.objectives import A2CLoss
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule

from environment.custom_env import SmartLearnEnv


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

DEFAULT_CONFIG = {
    "episodes": 100,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "entropy_coeff": 0.01,
    "hidden_size": 128,
    "max_grad_norm": 1.0,
    "seed": 42,
}

OBSERVATION_SIZE = 11
ACTION_SIZE = 7

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models" / "a2c"
LOG_DIR = PROJECT_ROOT / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. REPRODUCIBILITY
# ---------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------
# 3. ACTOR NETWORK
# ---------------------------------------------------------

def build_actor(action_spec, hidden_size):

    policy_network = nn.Sequential(
        nn.Linear(OBSERVATION_SIZE, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, ACTION_SIZE),
    )

    policy_module = TensorDictModule(
        policy_network,
        in_keys=["observation"],
        out_keys=["logits"],
    )

    actor = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        out_keys=["action"],
        spec=action_spec,
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )

    return actor


# ---------------------------------------------------------
# 4. CRITIC NETWORK
# ---------------------------------------------------------

def build_critic(hidden_size):

    value_network = nn.Sequential(
        nn.Linear(OBSERVATION_SIZE, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, 1),
    )

    critic = ValueOperator(
        module=value_network,
        in_keys=["observation"],
    )

    return critic


# ---------------------------------------------------------
# 5. COLLECT ONE EPISODE
# ---------------------------------------------------------

def collect_episode(env, actor, seed):

    observation, _ = env.reset(seed=seed)

    initial_mastery = float(np.mean(observation[:4]))

    transitions = []
    episode_reward = 0.0
    successes = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):

        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
        ).unsqueeze(0)

        tensordict = TensorDict(
            {"observation": observation_tensor},
            batch_size=[1],
        )

        with torch.no_grad():
            tensordict = actor(tensordict)

        action_tensor = tensordict.get("action")[0]
        action = int(torch.argmax(action_tensor).item())

        next_observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        done = terminated or truncated

        transition = TensorDict(
            {
                "observation": torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                ),
                "action": action_tensor.detach().clone(),
                "action_log_prob": tensordict.get(
                    "action_log_prob"
                )[0].detach().clone(),
                "next": {
                    "observation": torch.as_tensor(
                        next_observation,
                        dtype=torch.float32,
                    ),
                    "reward": torch.tensor(
                        [reward],
                        dtype=torch.float32,
                    ),
                    "done": torch.tensor(
                        [done],
                        dtype=torch.bool,
                    ),
                    "terminated": torch.tensor(
                        [terminated],
                        dtype=torch.bool,
                    ),
                },
            },
            batch_size=[],
        )

        transitions.append(transition)

        episode_reward += float(reward)

        if info.get("success", False):
            successes += 1

        observation = next_observation

    trajectory = TensorDict.stack(transitions, dim=0)

    final_mastery = float(
        info.get("mean_mastery", np.mean(observation[:4]))
    )

    episode_metrics = {
        "episode_reward": episode_reward,
        "mastery_improvement": final_mastery - initial_mastery,
        "final_mastery": final_mastery,
        "success_rate": successes / max(len(transitions), 1),
        "steps": len(transitions),
        "final_confidence": float(info.get("confidence", 0.0)),
        "final_engagement": float(info.get("engagement", 0.0)),
        "final_fatigue": float(info.get("fatigue", 0.0)),
    }

    return trajectory, episode_metrics


# ---------------------------------------------------------
# 6. TRAIN A2C
# ---------------------------------------------------------

def train_a2c(config=None, run_name="a2c_baseline"):

    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        config = {**DEFAULT_CONFIG, **config}

    set_seed(config["seed"])

    print("\n" + "=" * 60)
    print("SMARTLEARN GHANA - A2C TRAINING")
    print("=" * 60)

    for key, value in config.items():
        print(f"{key}: {value}")

    env = SmartLearnEnv()
    torchrl_env = GymWrapper(SmartLearnEnv())

    actor = build_actor(
        action_spec=torchrl_env.action_spec,
        hidden_size=config["hidden_size"],
    )

    critic = build_critic(
        hidden_size=config["hidden_size"],
    )

    loss_module = A2CLoss(
        actor_network=actor,
        critic_network=critic,
        entropy_bonus=True,
        entropy_coeff=config["entropy_coeff"],
        critic_coeff=1.0,
        loss_critic_type="smooth_l1",
    )

    # Generalized Advantage Estimation.
    gae = GAE(
        gamma=config["gamma"],
        lmbda=config["gae_lambda"],
        value_network=critic,
    )

    optimizer = torch.optim.Adam(
        loss_module.parameters(),
        lr=config["learning_rate"],
    )

    training_history = []

    print("\nStarting training...\n")

    for episode in range(1, config["episodes"] + 1):

        trajectory, metrics = collect_episode(
            env=env,
            actor=actor,
            seed=config["seed"] + episode,
        )

        # Estimate advantages and value targets.
        with torch.no_grad():
            trajectory = gae(trajectory)

        # A2C performs one update using the collected episode.
        loss_values = loss_module(trajectory)

        actor_loss = loss_values["loss_objective"]
        critic_loss = loss_values["loss_critic"]
        entropy_loss = loss_values["loss_entropy"]

        total_loss = (
            actor_loss
            + critic_loss
            + entropy_loss
        )

        optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            loss_module.parameters(),
            config["max_grad_norm"],
        )

        optimizer.step()

        metrics["episode"] = episode
        metrics["total_loss"] = float(total_loss.detach().item())
        metrics["actor_loss"] = float(actor_loss.detach().item())
        metrics["critic_loss"] = float(critic_loss.detach().item())
        metrics["entropy_loss"] = float(entropy_loss.detach().item())

        training_history.append(metrics)

        if episode % 10 == 0 or episode == 1:

            recent = training_history[-10:]

            print(
                f"Episode {episode:3d}/{config['episodes']} | "
                f"Reward: {np.mean([x['episode_reward'] for x in recent]):7.3f} | "
                f"Mastery: {np.mean([x['final_mastery'] for x in recent]):.3f} | "
                f"Success: {np.mean([x['success_rate'] for x in recent]):.2%} | "
                f"Actor loss: {metrics['actor_loss']:.4f} | "
                f"Critic loss: {metrics['critic_loss']:.4f} | "
                f"Entropy loss: {metrics['entropy_loss']:.4f}"
            )

    # -----------------------------------------------------
    # 7. SAVE MODEL
    # -----------------------------------------------------

    checkpoint_path = MODEL_DIR / f"{run_name}.pt"

    torch.save(
        {
            "loss_module_state_dict": loss_module.state_dict(),
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "config": config,
            "training_history": training_history,
        },
        checkpoint_path,
    )

    # -----------------------------------------------------
    # 8. SAVE TRAINING HISTORY
    # -----------------------------------------------------

    csv_path = LOG_DIR / f"{run_name}_history.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=training_history[0].keys(),
        )

        writer.writeheader()
        writer.writerows(training_history)

    # -----------------------------------------------------
    # 9. FINAL SUMMARY
    # -----------------------------------------------------

    final_episodes = training_history[-10:]

    print("\n" + "=" * 60)
    print("A2C TRAINING COMPLETED")
    print("=" * 60)

    print(
        "Average reward (final 10 episodes):",
        round(
            np.mean([
                item["episode_reward"]
                for item in final_episodes
            ]),
            4,
        ),
    )

    print(
        "Average final mastery (final 10 episodes):",
        round(
            np.mean([
                item["final_mastery"]
                for item in final_episodes
            ]),
            4,
        ),
    )

    print(
        "Average success rate (final 10 episodes):",
        round(
            np.mean([
                item["success_rate"]
                for item in final_episodes
            ]),
            4,
        ),
    )

    print(f"\nModel saved to: {checkpoint_path}")
    print(f"Training history saved to: {csv_path}")

    env.close()
    torchrl_env.close()

    return training_history


# ---------------------------------------------------------
# 10. RUN TRAINING
# ---------------------------------------------------------

if __name__ == "__main__":

    train_a2c(
        config=DEFAULT_CONFIG,
        run_name="a2c_baseline",
    )
