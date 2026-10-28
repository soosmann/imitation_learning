"""Evaluation utilities for Push-T policies."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from data import Normalizer
from model import BasePolicy
from plot_training import plot_training

ENV_ID = "gym_pusht/PushT-v0"
NUM_EVAL_EPISODES = 100


class Logger:
    """Logger for logging metrics."""

    def __init__(self, path: Path):
        """
        Create a new directory for experiment metrics and artifacts.

        Params:
            path (`Path`): The experiment directory to create.

        Raises:
            FileExistsError: If the directory already exists.
        """
        if path.exists():
            raise FileExistsError(f"Log directory {path} already exists.")
        path.mkdir(parents=True)
        self.path = path
        self.csv_path = path / "log.csv"
        self.header = ["step"]
        self.rows: dict[int, dict[str, int | float]] = {}

    def log(self, row: dict[str, Any], step: int) -> None:
        """
        Merge numeric metrics for a step and rewrite the experiment CSV.

        Params:
            row (`dict[str, Any]`): Metrics to log. Non-numeric values are ignored.
            step (`int`): The training step associated with these metrics.

        Returns:
            None: Updates `log.csv` when numeric metrics are provided.
        """
        # Merge training and evaluation metrics into one row per step.
        # Videos are saved separately and do not belong in the metrics CSV.
        metrics = {}
        for key, value in row.items():
            if key == "step":
                continue
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                value = value.detach().cpu().item()
            if isinstance(value, (int, float, np.number)):
                metrics[key] = float(value)
        if not metrics:
            return
        for key in metrics:
            if key not in self.header:
                self.header.append(key)
        self.rows.setdefault(step, {"step": step}).update(metrics)
        with self.csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writeheader()
            writer.writerows(self.rows.values())


def resize_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """
    Resizing a frame (image of a video) to the selected `size`.

    Params:
        frame (`np.ndarray`): The frame to be resized.
        size (`tuple[int, int]`): New frame dimensions in format `(width, height)`.

    Returns:
        np.ndarray: The resized frame.
    """
    image = Image.fromarray(frame)
    resized = image.resize(size, resample=Image.BILINEAR)
    return np.asarray(resized)


def encode_video(
    frames: list[np.ndarray],
    fps: int = 20,
) -> io.BytesIO | None:
    """
    Creates a video out of a list of frames.

    Params:
        frames (`list[np.ndarray]`): The list of frames to be concluded to a video.
        fps (`int`): Frames per second of the video.

    Returns:
        io.BytesIO | None: The MP4 video buffer, or None if no frames were provided.
    """
    if not frames:
        return None

    buffer = io.BytesIO()

    with imageio.get_writer(
        buffer,
        format="mp4",
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)

    buffer.seek(0)
    return buffer


def save_best_checkpoint(
    model: BasePolicy,
    mean_reward: float,
    log_dir: Path,
) -> bool:
    """
    Save the policy if its mean evaluation reward improves.

    Params:
        model (`BasePolicy`): The model to be saved.
        mean_reward (`float`): The mean reward achieved during eval.
        log_dir (`Path`): Dir of logging, model will be saved there.

    Returns:
        bool: Whether a new best model was saved.

    """
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = log_dir / "best_model.pt"
    best_reward_path = log_dir / "best_reward.txt"

    best_reward = (
        float(best_reward_path.read_text())
        if best_reward_path.exists()
        else float("-inf")
    )

    if mean_reward <= best_reward:
        return False

    # Save to a temporary file first to avoid leaving a partial checkpoint
    tmp_path = log_dir / "best_model.tmp"
    torch.save(model.state_dict(), tmp_path)
    tmp_path.replace(checkpoint_path)

    best_reward_path.write_text(str(mean_reward))
    return True


def evaluate_policy(
    model: BasePolicy,
    normalizer: Normalizer,
    device: torch.device,
    chunk_size: int,
    video_size: tuple[int, int],
    num_video_episodes: int,
    video_every_n_steps: int,
    flow_num_steps: int,
    step: int,
    logger: Logger,
) -> None:
    """
    Evaluate a policy in the Push-T environment and log results.

    This function runs a fixed number of evaluation episodes in the Push-T gym
    environment using the provided policy. It normalizes observations with the
    given normalizer, requests a chunk of actions from the policy (optionally
    using multiple sampling steps for flow-based policies), and executes those
    actions in the environment until each episode terminates.

    Metrics:
        - Logs the mean of per-episode maximum reward.
        - Refreshes the training loss and evaluation reward plots.
        - Optionally logs rendered rollout videos for the first
          ``num_video_episodes`` episodes.

    Checkpointing:
        - Saves the policy weights when mean evaluation reward improves.

    Params:
        model (`BasePolicy`): The policy to evaluate.
        normalizer (`Normalizer`): Normalizer used to scale states and actions.
        device (`torch.device`): Device on which to run policy inference.
        chunk_size (`int`): Number of actions to generate per policy call.
        video_size (`tuple[int, int]`): (width, height) for rendered rollout videos.
        num_video_episodes (`int`): How many episodes to record videos for.
        video_every_n_steps (`int`): How often a video should be saved.
        flow_num_steps (`int`): Number of denoising steps used by flow policies.
        step (`int`): Training step used for logging and artifact metadata.
        logger (`Logger`): Logger for logging metrics.

    Returns:
        None: Logs metrics, updates plots, and saves selected videos and checkpoints.
    """
    model.eval()
    rewards: list[float] = []
    videos: list = []

    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low
    action_high = env.action_space.high

    for ep_idx in range(NUM_EVAL_EPISODES):
        obs, _ = env.reset(seed=ep_idx)
        done = False
        chunk_index = chunk_size
        action_chunk: np.ndarray | None = None
        frames: list[np.ndarray] = []
        max_reward = 0.0
        save_videos = video_every_n_steps > 0 and step % video_every_n_steps == 0

        video_dir = logger.path / "videos"
        if save_videos:
            video_dir.mkdir(parents=True, exist_ok=True)

        save_video = save_videos and ep_idx < num_video_episodes

        while not done:
            if action_chunk is None or chunk_index >= chunk_size:
                state = (
                    torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
                )
                with torch.no_grad():
                    pred_chunk = (
                        model.sample_actions(
                            state.unsqueeze(0), num_steps=flow_num_steps
                        )
                        .cpu()
                        .numpy()[0]
                    )
                action_chunk = normalizer.denormalize_action(pred_chunk)
                action_chunk = np.clip(action_chunk, action_low, action_high)
                chunk_index = 0

            action = action_chunk[chunk_index]
            obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
            if save_video:
                frame = env.render()
                frame = resize_frame(frame, video_size)
                frames.append(frame)
            max_reward = max(max_reward, float(reward))
            done = terminated or truncated
            chunk_index += 1

        rewards.append(max_reward)
        if save_video:
            video = encode_video(frames, fps=20)
            if video is not None:
                videos.append(video)
                video_path = video_dir / f"eval_step_{step}_ep{ep_idx}.mp4"
                video_path.write_bytes(video.getvalue())

    env.close()

    mean_reward = float(np.mean(rewards))

    log_data: dict[str, float | io.BytesIO] = {
        "eval/mean_reward": mean_reward,
    }

    for idx, video in enumerate(videos):
        log_data[f"eval/rollout_ep{idx}"] = video

    logger.log(log_data, step=step)
    plot_training(logger.csv_path)

    if save_best_checkpoint(
        model=model,
        mean_reward=mean_reward,
        log_dir=logger.path,
    ):
        print(f"New best model at step {step}: mean reward = {mean_reward:.4f}")
