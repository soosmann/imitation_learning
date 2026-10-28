"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gym_pusht  # needed for gymnasium usage

# from typing import Any
import numpy as np
import torch
import tyro
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from evaluation import Logger, evaluate_policy
from model import PolicyType, build_policy

LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    """Configuration for Push-T training, evaluation, and local logging."""

    # the path to download the Push-T dataset to.
    data_dir: Path = Path("data")
    # the policy type - either MSE or flow.
    policy_type: PolicyType = "flow"  # mse flow
    # the number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    # the action chunk size.
    chunk_size: int = 8
    # amount of similarly processed samples
    batch_size: int = 128
    # learning rate
    lr: float = 3e-4
    # regularization using weight decay in adam optim
    weight_decay: float = 0.0
    # sizes of hidden layers in mlp policy
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # the number of epochs to train for.
    num_epochs: int = 400
    # how often to run evaluation, measured in epochs.
    eval_interval: int = 10
    # amount of videos captured during eval when videos are captured
    num_video_episodes: int = 3
    # pixel dims of video
    video_size: tuple[int, int] = (256, 256)
    # every n steps take the videos, must coincide with eval_interval
    video_every_n_steps: int = 50
    # random seed for reproducibiliy
    seed: int = 42
    # experiment name suffix for logging
    exp_name: str | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    """
    Creation of a full training config for the Push-T scenario.

    Params:
        args (`list[str] | None`): Training arguments, or None to read the command line.
        defaults (`TrainConfig | None`): Default values to override, or None for standard defaults.
        description (`str`): Description of the training case.

    Returns:
        TrainConfig: The parsed training configuration.
    """
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    """
    Setting seeds for reproducibility.

    Params:
        seed (`int`): Seed value.

    Returns:
        None: Does only perform setting of seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_training(config: TrainConfig) -> None:
    """
    Runs the training of the Push-T case using the config provided.

    Params:
        config (`TrainConfig`): The config.

    Returns:
        None: Logs training metrics and saves evaluation artifacts and best weights.
    """
    # configuration and device selection
    set_seed(config.seed)
    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # dataset prep
    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # print(next(iter(loader)))

    # build model
    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)

    # training prep
    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    logger = Logger(log_dir)

    # Save run metadata before training, including runs without evaluation.
    config_values = asdict(config)
    config_values["data_dir"] = str(config.data_dir)
    (log_dir / "config.json").write_text(
        json.dumps(config_values, indent=2) + "\n", encoding="utf-8"
    )
    np.savez(log_dir / "normalizer.npz", **asdict(normalizer))

    optim = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # actual training
    for i in range(config.num_epochs):
        epoch = i + 1
        losses = []
        for states, actions in loader:
            states, actions = states.to(device), actions.to(device)
            model.zero_grad()

            loss = model.compute_loss(states, actions)
            losses.append(loss.detach().cpu())

            loss.backward()

            optim.step()
        epoch_loss = sum(losses) / len(losses)
        logger.log({"epoch_loss": epoch_loss}, step=epoch)
        print(f"{epoch}.: {epoch_loss}")

        if epoch % config.eval_interval == 0:
            evaluate_policy(
                model=model,
                normalizer=normalizer,
                device=device,
                chunk_size=config.chunk_size,
                video_size=config.video_size,
                num_video_episodes=config.num_video_episodes,
                video_every_n_steps=config.video_every_n_steps,
                flow_num_steps=config.flow_num_steps,
                step=epoch,
                logger=logger,
            )


def main() -> None:
    """
    Parse command-line arguments and start training.

    Returns:
        None: Runs training with the parsed configuration.
    """
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
