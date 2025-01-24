from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions.normal import Normal


class CriticNetwork(nn.Module):
    """
    Critic network used in SAC (Soft Actor Critic).
    """

    def __init__(
        self,
        beta: float,
        input_dims: Tuple[int, ...],
        n_actions: int,
        hidden_dims: Tuple[int, ...],
        name: str,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Critic network used in SAC (Soft Actor Critic).

        Args:
            beta (float): Critic learning rate.
            input_dims (Tuple[int, ...]): The dimensions of the obversevation space.
            n_actions (int): The number of actions.
            hidden_dims (Tuple[int, ...]): Hidden layer dimensions
            name (str): Name used for checkpointing.
            device (str, optional): The device to load onto. Defaults to "cpu".
            dtype (torch.dtype, optional): THe data type of the tensors.
                Defaults to torch.float32.
        """
        super(CriticNetwork, self).__init__()

        # Store parameters
        self.beta = beta
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims
        self.name = name

        self.dtype = dtype

        # Define layers
        self.input = nn.Linear(
            in_features=self.input_dims[0] + n_actions,
            out_features=self.hidden_dims[0],
            dtype=self.dtype,
        )

        # Define hidden layers
        self.hidden_layers: List[nn.Linear] = []
        for dims_idx in range(len(self.hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Linear(
                    in_features=self.hidden_dims[dims_idx],
                    out_features=self.hidden_dims[dims_idx + 1],
                    dtype=self.dtype,
                )
            )

        # Define output layer
        self.q = nn.Linear(
            in_features=self.hidden_dims[-1],
            out_features=1,
            dtype=self.dtype,
        )

        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)

        # Define device
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.DEVICE)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the critic network.

        Args:
            state (torch.Tensor): Current state.
            action (torch.Tensor): Current action.

        Returns:
            torch.Tensor: The Q action value associated with the state and action.
        """
        # Concatenate the state and action and pass through the input layer.
        action_value = self.input(torch.cat([state, action]), dim=1)
        action_value = F.relu(action_value)  # TODO: Make this inplace

        # Pass through the hidden layers
        for layer in self.hidden_layers:
            action_value = layer(action_value)
            action_value = F.relu(action_value)

        # Pass through the q action value layer.
        q = self.q(action_value)

        return q

    def save_checkpoint(self): ...

    def load_checkpoint(self): ...


class ValueNetwork(nn.Module):
    def __init__(
        self,
        beta: float,
        input_dims: Tuple[int, ...],
        hidden_dims: Tuple[int, ...],
        name: str,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Value network used in SAC (Soft Actor Critic).

        Args:
            beta (float): The learning rate of the value network.
            input_dims (Tuple[int, ...]): The dimensions of the observation space.
            hidden_dims (Tuple[int, ...]): The dimensions of the hidden layers.
            name (str): The name used for checkpointing.
            dtype (torch.dtype, optional): The datatype of the network.
                Defaults to torch.float32.
        """
        super(ValueNetwork, self).__init__()

        # Store parameters
        self.beta = beta
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.name = name

        self.dtype = dtype

        # Define layers
        self.input = nn.Linear(
            in_features=self.input_dims[0],
            out_features=self.hidden_dims[0],
            dtype=self.dtype,
        )

        # Define hidden layers
        self.hidden_layers: List[nn.Linear] = []
        for dims_idx in range(len(self.hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Linear(
                    in_features=self.hidden_dims[dims_idx],
                    out_features=self.hidden_dims[dims_idx + 1],
                    dtype=self.dtype,
                )
            )

        # Define output layer
        self.v = nn.Linear(
            in_features=self.hidden_dims[-1],
            out_features=1,
            dtype=self.dtype,
        )

        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)

        # Define and move to device
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.DEVICE)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the value network.

        Args:
            state (torch.Tensor): The current state.

        Returns:
            torch.Tensor: The value associated with the state.
        """
        # Pass through the input layer
        state_value = self.input(state)
        state_value = F.relu(state_value)

        # Pass through the hidden layers
        for layer in self.hidden_layers:
            state_value = layer(state_value)
            state_value = F.relu(state_value)

        # Pass through the value layer
        v = self.v(state_value)

        return v

    def save_checkpoint(self): ...

    def load_checkpoint(self): ...


class ActorNetwork(nn.Module):
    def __init__(
        self,
        alpha: int,
        input_dims: Tuple[int, ...],
        max_action: float,
        n_actions: int,
        hidden_dims: Tuple[int, ...],
        name: str,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Actor network

        Args:
            alpha (int): Actor learning rate.
            input_dims (Tuple[int, ...]): Input dimensions.
            max_action (int): Maximum action value.
            n_actions (int): Number of actions.
            hidden_dims (Tuple[int, ...]): Hidden layer dimensions.
            name (str): Name used for checkpointing.
            dtype (torch.dtype, optional): Tensor data type. Defaults to torch.float32.
        """
        super(ActorNetwork, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # store parameters
        self.alpha = alpha
        self.input_dims = input_dims
        self.max_action = torch.tensor(max_action).to(self.device)
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims

        self.name = name
        self.dtype = dtype

        # used for the reparameterization trick
        self.reparimeterization_noise = 1e-6

        # define layers
        self.input = nn.Linear(
            in_features=self.input_dims[0],
            out_features=self.hidden_dims[0],
            dtype=self.dtype,
        )

        # define hidden layers
        self.hidden_layers: List[nn.Linear] = []
        for dims_idx in range(len(self.hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Linear(
                    in_features=self.hidden_dims[dims_idx],
                    out_features=self.hidden_dims[dims_idx + 1],
                    dtype=self.dtype,
                )
            )

        # define mu and sigma layers
        self.mu = nn.Linear(
            in_features=self.hidden_dims[-1],
            out_features=self.n_actions,
            dtype=self.dtype,
        )
        self.sigma = nn.Linear(
            in_features=self.hidden_dims[-1],
            out_features=self.n_actions,
            dtype=self.dtype,
        )

        # define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

        # define and move to device
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the actor network.

        Args:
            state (torch.Tensor): The current state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The mean and standard deviation of the action distribution.
        """
        # pass through the input layer
        prob = self.input(state)
        prob = F.relu(prob)

        # pass through the hidden layers
        for layer in self.hidden_layers:
            prob = layer(prob)
            prob = F.relu(prob)

        # pass through the mu and sigma layers
        mu = self.mu(prob)
        sigma = self.sigma(prob)

        # clamp sigma to avoid numerical instability.
        sigma = torch.clamp(sigma, min=self.reparimeterization_noise, max=1)

        return mu, sigma

    def sample_normal(
        self, state: torch.Tensor, reparameterize: bool = True
    ) -> Tuple[torch.Tensor]:
        """
        Sample from a normal distribution and reparameterize.

        Args:
            state (torch.Tensor): The current state.
            reparameterize (bool, optional): Whether to reparameterize the distribution.
                Defaults to True.

        Returns:
            Tuple[torch.Tensor]: The sampled actions and the log probability of the
                sampled actions.
        """
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = torch.tanh(actions) * self.max_action

        log_probs: torch.Tensor
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.reparimeterization_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self): ...

    def load_checkpoint(self): ...
