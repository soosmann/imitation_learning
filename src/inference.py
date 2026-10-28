"""Load a trained Push-T policy and run inference episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import tyro

from data import Normalizer
from evaluation import ENV_ID, encode_video, resize_frame
from model import BasePolicy, build_policy
from train import TrainConfig, set_seed


@dataclass
class InferenceConfig:
    """Configuration for loading a trained policy and running inference."""

    # experiment directory containing best_model.pt, config.json, and normalizer.npz.
    exp_dir: Path
    # number of episodes to run.
    num_episodes: int = 1
    # initial environment seed; each following episode increments it by one.
    seed: int = 0
    # device for inference, using the training device preference when set to auto.
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    # show the rollout in a window. Cannot be combined with video recording.
    render: bool = False
    # optional directory for saving one MP4 per episode.
    video_dir: Path | None = None


def load_policy(
    exp_dir: Path,
    device: torch.device,
) -> tuple[BasePolicy, Normalizer, TrainConfig]:
    """
    Reconstruct a policy and its normalizer from a saved experiment.

    Params:
        exp_dir (`Path`): Directory containing weights, configuration, and statistics.
        device (`torch.device`): Device on which to load the policy.

    Returns:
        tuple[BasePolicy, Normalizer, TrainConfig]: Policy in evaluation mode,
            restored normalizer, and the original training configuration.

    Raises:
        FileNotFoundError: If any of the required experiment files are missing.
    """
    for filename in ("best_model.pt", "config.json", "normalizer.npz"):
        if not (exp_dir / filename).is_file():
            raise FileNotFoundError(
                f"Missing {exp_dir / filename}. Inference requires weights and "
                "the configuration and normalization statistics saved during training."
            )

    values = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
    values["data_dir"] = Path(values["data_dir"])
    values["hidden_dims"] = tuple(values["hidden_dims"])
    values["video_size"] = tuple(values["video_size"])
    train_config = TrainConfig(**values)

    with np.load(exp_dir / "normalizer.npz", allow_pickle=False) as statistics:
        normalizer = Normalizer(
            state_mean=statistics["state_mean"],
            state_std=statistics["state_std"],
            action_mean=statistics["action_mean"],
            action_std=statistics["action_std"],
        )

    model = build_policy(
        train_config.policy_type,
        state_dim=normalizer.state_mean.size,
        action_dim=normalizer.action_mean.size,
        chunk_size=train_config.chunk_size,
        hidden_dims=train_config.hidden_dims,
    ).to(device)
    weights = torch.load(
        exp_dir / "best_model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(weights)
    model.eval()
    return model, normalizer, train_config


@torch.no_grad()
def run_inference(config: InferenceConfig) -> list[float]:
    """
    Run a saved policy in Push-T and report the maximum reward of each episode.

    Params:
        config (`InferenceConfig`): Loading, rollout, and optional recording settings.

    Returns:
        list[float]: The maximum reward reached in each episode.

    Raises:
        ValueError: If the episode count is invalid or rendering and recording are combined.
    """
    if config.num_episodes < 1:
        raise ValueError("num_episodes must be at least 1")
    if config.render and config.video_dir is not None:
        raise ValueError("Choose either live rendering or video recording")

    set_seed(config.seed)
    if config.device != "auto":
        device = torch.device(config.device)
    elif torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model, normalizer, train_config = load_policy(config.exp_dir, device)
    if config.video_dir is not None:
        config.video_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        ENV_ID,
        obs_type="state",
        render_mode="human" if config.render else "rgb_array",
    )
    rewards: list[float] = []
    try:
        for ep_idx in range(config.num_episodes):
            episode_seed = config.seed + ep_idx
            obs, _ = env.reset(seed=episode_seed)
            done = False
            max_reward = 0.0
            frames: list[np.ndarray] = []

            while not done:
                state = (
                    torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
                )
                predicted = (
                    model.sample_actions(
                        state.unsqueeze(0), num_steps=train_config.flow_num_steps
                    )
                    .cpu()
                    .numpy()[0]
                )
                actions = normalizer.denormalize_action(predicted)
                actions = np.clip(actions, env.action_space.low, env.action_space.high)

                # Execute the full chunk, stopping immediately when the episode ends.
                for action in actions:
                    obs, reward, terminated, truncated, _ = env.step(
                        action.astype(np.float32)
                    )
                    max_reward = max(max_reward, float(reward))
                    if config.video_dir is not None:
                        frames.append(
                            resize_frame(env.render(), train_config.video_size)
                        )
                    done = terminated or truncated
                    if done:
                        break

            rewards.append(max_reward)
            print(
                f"Episode {ep_idx + 1}, seed {episode_seed}: max reward = {max_reward:.4f}"
            )
            if config.video_dir is not None:
                video = encode_video(frames, fps=env.metadata.get("render_fps", 20))
                if video is not None:
                    video_path = (
                        config.video_dir / f"episode_{ep_idx}_seed_{episode_seed}.mp4"
                    )
                    video_path.write_bytes(video.getvalue())
                    print(f"Saved video: {video_path}")
    finally:
        env.close()

    print(f"Mean episode maximum reward: {np.mean(rewards):.4f}")
    return rewards


def main() -> None:
    """
    Parse command-line arguments and run a saved policy.

    Returns:
        None: Runs inference and prints episode rewards.
    """
    config = tyro.cli(InferenceConfig, description=__doc__)
    run_inference(config)


if __name__ == "__main__":
    main()
