"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        """
        Initialize the dimensions shared by action chunking policies.

        Params:
            state_dim (`int`): The number of state features.
            action_dim (`int`): The number of features in one action.
            chunk_size (`int`): The number of actions predicted per policy call.
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute training loss for a batch.

        Params:
            state (`torch.Tensor`): Input states with shape (batch, state_dim).
            action_chunk (`torch.Tensor`): Targets with shape (batch, chunk_size, action_dim).

        Returns:
            torch.Tensor: The scalar training loss.
        """

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """
        Generate a chunk of actions for each input state.

        Params:
            state (`torch.Tensor`): Input states with shape (batch, state_dim).
            num_steps (`int`): The number of integration steps for flow policies.

        Returns:
            torch.Tensor: Actions with shape (batch, chunk_size, action_dim).
        """


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        """
        Build an MLP that predicts an action chunk directly from a state.

        Params:
            state_dim (`int`): The number of state features.
            action_dim (`int`): The number of features in one action.
            chunk_size (`int`): The number of actions predicted per policy call.
            hidden_dims (`tuple[int, ...]`): The width of each hidden layer.
        """
        super().__init__(state_dim, action_dim, chunk_size)

        # The network predicts every action in the chunk at once.
        output_dim = chunk_size * action_dim

        layers = []
        in_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes MSE loss using states for prediction and comparing them against the target.

        Params:
            state (`torch.Tensor`): The input states.
            action_chunk (`torch.Tensor`): The target actions.

        Returns:
            torch.Tensor: The mse loss value.
        """
        # get preds for batch in format (N, chunk_size * action_dim)
        preds = self.network(state)

        # reshape to (N, chunk_size, action_dim)
        preds_actions = preds.reshape(-1, self.chunk_size, self.action_dim)

        # mse loss calculation
        return F.mse_loss(preds_actions, action_chunk)

    @torch.no_grad()
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Based on state(s), provide the next action chunk. `num_steps` is ignored
        because this policy predicts actions directly.

        Params:
            state (`torch.Tensor`): The state(s) to produce new actions for.
            num_steps (`int`): Integration steps for flow policies, ignored here.

        Returns:
            torch.Tensor: Actions with shape (batch, chunk_size, action_dim).
        """
        # get preds for batch in format (N, chunk_size * action_dim)
        preds = self.network(state)

        # return action in shape (N, chunk_size, action_dim)
        return preds.reshape(-1, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        """
        Build an MLP that predicts the velocity of a noisy action chunk.

        Params:
            state_dim (`int`): The number of state features.
            action_dim (`int`): The number of features in one action.
            chunk_size (`int`): The number of actions generated per policy call.
            hidden_dims (`tuple[int, ...]`): The width of each hidden layer.
        """
        super().__init__(state_dim, action_dim, chunk_size)

        self.output_dim = chunk_size * action_dim

        # Condition the vector field on state, noisy action chunk, and time.
        layers = []
        in_dim = state_dim + self.output_dim + 1

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, self.output_dim))
        self.network = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute flow matching loss along a linear path from noise to expert actions.

        Params:
            state (`torch.Tensor`): Input states with shape (batch, state_dim).
            action_chunk (`torch.Tensor`): Targets with shape (batch, chunk_size, action_dim).

        Returns:
            torch.Tensor: The MSE between predicted and target velocities.
        """
        batch_size = state.shape[0]

        # Flatten the expert action chunks: (B, H, A) -> (B, H*A).
        target = action_chunk.reshape(batch_size, self.output_dim)

        # Sample Gaussian noise and interpolation time.
        noise = torch.randn_like(target)
        t = torch.rand(batch_size, 1, device=target.device, dtype=target.dtype)

        # Linear probability path: noise at t=0, expert action at t=1.
        t_broadcast = t
        x_t = (1.0 - t_broadcast) * noise + t_broadcast * target

        # Constant target velocity along the linear path.
        target_velocity = target - noise

        # Predict the velocity conditioned on state, x_t, and time.
        model_input = torch.cat([state, x_t, t], dim=-1)
        predicted_velocity = self.network(model_input)

        return F.mse_loss(predicted_velocity, target_velocity)

    @torch.no_grad()
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Generate action chunks by integrating the learned velocity field from noise.

        Params:
            state (`torch.Tensor`): Input states with shape (batch, state_dim).
            num_steps (`int`): The number of Euler integration steps, at least one.

        Returns:
            torch.Tensor: Actions with shape (batch, chunk_size, action_dim).

        Raises:
            ValueError: If `num_steps` is less than one.
        """
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")

        batch_size = state.shape[0]

        # Start from Gaussian noise.
        x = torch.randn(
            batch_size,
            self.output_dim,
            device=state.device,
            dtype=state.dtype,
        )

        # Integrate dx/dt = v_theta(state, x, t) from t=0 to t=1.
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full(
                (batch_size, 1),
                i * dt,
                device=state.device,
                dtype=state.dtype,
            )

            model_input = torch.cat([state, x, t], dim=-1)
            velocity = self.network(model_input)

            # Euler integration.
            x = x + dt * velocity

        return x.reshape(batch_size, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    """
    Create the selected action chunking policy.

    Params:
        policy_type (`PolicyType`): Policy to build, either `mse` or `flow`.
        state_dim (`int`): The number of state features.
        action_dim (`int`): The number of features in one action.
        chunk_size (`int`): The number of actions generated per policy call.
        hidden_dims (`tuple[int, ...]`): The width of each hidden layer.

    Returns:
        BasePolicy: The initialized MSE or flow matching policy.

    Raises:
        ValueError: If the policy type is unknown.
    """
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
