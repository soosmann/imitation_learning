"""Dataset utilities for Push-T."""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
ZARR_RELATIVE_PATH = Path("pusht") / "pusht_cchi_v7_replay.zarr"


@dataclass(frozen=True)
class Normalizer:
    """Feature-wise normalizer for states and actions."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Keep standard deviations above a minimum value to avoid division by zero.

        Params:
            std (`np.ndarray`): The feature-wise standard deviations.
            eps (`float`): The minimum allowed standard deviation.

        Returns:
            np.ndarray: The standard deviations with the minimum applied.
        """
        return np.maximum(std, eps)

    @classmethod
    def from_data(cls, states: np.ndarray, actions: np.ndarray) -> Normalizer:
        """
        Retrieves the mean and standard deviation of states and actions and saves them as `np.ndarray`s.

        Params:
            states (`np.ndarray`): All states of the dataset.
            actions (`np.ndarray`): All actions of the dataset.

        Returns:
            Normalizer: An instance of the class with calculated mean and std.
        """
        state_mean = states.mean(axis=0)
        state_std = cls._safe_std(states.std(axis=0))
        action_mean = actions.mean(axis=0)
        action_std = cls._safe_std(actions.std(axis=0))
        return cls(state_mean, state_std, action_mean, action_std)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalization of state.

        Params:
            state (`np.ndarray`): The state to be normalized.

        Returns:
            np.ndarray: The normalized state.
        """
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalization of action.

        Params:
            action (`np.ndarray`): The action to be normalized.

        Returns:
            np.ndarray: The normalized action.
        """
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Denormalization of action.

        Params:
            action (`np.ndarray`): The action to be denormalized.

        Returns:
            np.ndarray: The denormalized action.
        """
        return action * self.action_std + self.action_mean


def download_pusht(dataset_dir: Path) -> Path:
    """
    Download and extract the Push-T dataset if needed.
    Returns the path to the extracted Zarr dataset.

    Params:
        dataset_dir (`Path`): The path to the dir where to extract the Push-T dataset.

    Returns:
        Path: The path to the dataset.
    """

    dataset_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = dataset_dir / ZARR_RELATIVE_PATH
    if zarr_path.exists():
        return zarr_path

    zip_path = dataset_dir / "pusht.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(PUSHT_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    return zarr_path


def load_pusht_zarr(zarr_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loading of the Push-T dataset.

    Params:
        zarr_path (`Path`): The path to the dataset.

    Returns:
        tuple: Three arrays, one for the states, one for the actions, one for the episode ends.
    """
    root = zarr.open(zarr_path, mode="r")
    states = np.asarray(root["data"]["state"][:], dtype=np.float32)
    actions = np.asarray(root["data"]["action"][:], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    return states, actions, episode_ends


def build_valid_indices(episode_ends: np.ndarray, chunk_size: int) -> np.ndarray:
    """
    Find starting indices for action chunks that stay within one episode.

    Params:
        episode_ends (`np.ndarray`): The exclusive end index of each episode.
        chunk_size (`int`): The number of actions in each chunk.

    Returns:
        np.ndarray: Valid starting indices, excluding episodes shorter than a chunk.
    """
    starts = np.concatenate(([0], episode_ends[:-1]))
    indices: list[int] = []
    for start, end in zip(starts, episode_ends, strict=True):
        last_start = end - chunk_size
        if last_start < start:
            continue
        indices.extend(range(start, last_start + 1))
    return np.asarray(indices, dtype=np.int64)


class PushtChunkDataset(Dataset):
    """Dataset of (state, action_chunk) pairs using a sliding window."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        chunk_size: int,
        normalizer: Normalizer | None = None,
    ) -> None:
        """
        Init of a Push-T dataset.

        Params:
            states (`np.ndarray`): All states in the dataset.
            actions (`np.ndarray`): All actions in the dataset.
            episode_ends (`np.ndarray`): All values for when an episode ends in the dataset.
            chunk_size (`int`): Size of the amount of steps calculated in one step.
            normalizer (`Normalizer | None`): Normalizer for normalization if wanted.
        """
        self.states = states
        self.actions = actions
        self.chunk_size = chunk_size
        self.normalizer = normalizer
        self.indices = build_valid_indices(episode_ends, chunk_size)

    def __len__(self) -> int:
        """
        Length of the dataset.

        Returns:
            int: The number of valid state and action chunk pairs.
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve an item based on its index in the dataset.

        Params:
            idx (`int`): The index of the sample.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The state and action chunk tensors.
        """
        t = int(self.indices[idx])
        state = self.states[t]
        action_chunk = self.actions[t : t + self.chunk_size]

        # normalize if normalizer provided
        if self.normalizer is not None:
            state = self.normalizer.normalize_state(state)
            action_chunk = self.normalizer.normalize_action(action_chunk)

        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(action_chunk).float(),
        )
