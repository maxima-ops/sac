from typing import Tuple

import torch


class ReplayBuffer:
    def __init__(
        self,
        max_size: int,
        input_shape: Tuple[int, ...],
        n_actions: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Experience Replay Buffer used for storing experience buffer to stabalise
        critic training.


        Args:
            max_size (int): The maximum size of the buffer.

            input_shape (Tuple[int, ...]): The input shape corisponding to the shape of
                the environment.
            n_actions (int): The number of actions or components of that action.
            dtype (torch.dtype, optional): THe data type of the tensors.
                Defaults to torch.float32.
        """
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.mem_size = max_size
        self.mem_counter = 0

        self.state_memory = torch.zeros(
            size=(self.mem_size, *input_shape), device=self.DEVICE, dtype=dtype
        )
        self.new_state_memory = torch.zeros(
            size=(self.mem_size, *input_shape), device=self.DEVICE, dtype=dtype
        )
        self.action_memory = torch.zeros(
            size=(self.mem_size, n_actions), device=self.DEVICE, dtype=dtype
        )
        self.reward_memory = torch.zeros(
            size=(self.mem_size, ), device=self.DEVICE, dtype=dtype
        )
        self.terminal_memory = torch.zeros(
            size=(self.mem_size, ), device=self.DEVICE, dtype=torch.bool
        )

    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        new_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """
        Store a transition in the replay buffer where:

        .. math::

            D = {(s_t, a_t, r_t, s_{t+1}, done)}

        Args:
            state (torch.Tensor): The current state of the environment.
            action (torch.Tensor): The action taken in the environment.
            reward (torch.Tensor): The reward received from the environment.
            new_state (torch.Tensor): The new state of the environment.
            done (torch.Tensor): The terminal state of the environment.
        """
        index = self.mem_counter % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def sample_buffer(
        self, mini_batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample experience from the replay buffer.


        Args:
            mini_batch_size (int): the size of the batch to sample. Needs to be
                less than the maximum size of the buffer.

        Returns:
            Tuple[ torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor ]:
                Samples from the buffer containing:

                .. math::

                    (s_t, a_t, r_t, s_{t+1}, done)
        """
        max_mem = min(self.mem_counter, self.mem_size)

        batch_indexes = torch.randint(low=0, high=max_mem, size=(mini_batch_size,))

        states = self.state_memory[batch_indexes]
        actions = self.action_memory[batch_indexes]
        rewards = self.reward_memory[batch_indexes]
        new_state = self.new_state_memory[batch_indexes]
        dones = self.terminal_memory[batch_indexes]

        return states, actions, rewards, new_state, dones
