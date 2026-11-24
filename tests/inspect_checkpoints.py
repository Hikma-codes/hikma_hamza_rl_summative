import torch
from pathlib import Path

checkpoints = {
    "DQN": "models/dqn/dqn_exp_02_low_learning_rate.pt",
    "REINFORCE": "models/reinforce/reinforce_exp_07_larger_network.pt",
    "PPO": "models/ppo/ppo_exp_05_lower_gae_lambda.pt",
    "A2C": "models/a2c/a2c_exp_05_higher_discount.pt",
}

for name, checkpoint_path in checkpoints.items():
    print("\n" + "=" * 55)
    print(f"{name} CHECKPOINT")
    print("=" * 55)

    path = Path(checkpoint_path)

    if not path.exists():
        print(f"File not found: {path}")
        continue

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    print("Type:", type(data).__name__)

    if isinstance(data, dict):
        print("Top-level keys:", list(data.keys()))

        for key, value in data.items():
            if isinstance(value, dict):
                print(f"  {key}: {list(value.keys())[:8]}")

            elif isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor {tuple(value.shape)}")

            else:
                print(f"  {key}: {type(value).__name__}")
