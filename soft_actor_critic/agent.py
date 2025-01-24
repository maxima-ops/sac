from typing import List, Optional

import torch
import torch.nn.functional as F

from gymnasium import Env

from soft_actor_critic.networks import ActorNetwork, CriticNetwork, ValueNetwork
from soft_actor_critic.replay_buffer import ReplayBuffer


class Agent:
    def __init__(
        self,
        alpha: float = 0.0003,
        beta: float = 0.0003,
        input_dims: List[int] = [8],
        hidden_dims: List[int] = [256, 256],
        n_actions: int = 2,
        batch_size: int = 256,
        max_size: int = 1000000,
        tau: float = 0.005,
        env: Env = Env(),
        gamma: float = 0.99,
        reward_scale: int = 2,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Agent class that contains the actor, critic and value networks as well as the
        replay buffer and optimizers for training the networks.

        Args:
            alpha (float, optional): The learning rate for the actor network.
                Defaults to 0.0003.
            beta (float, optional): The learning rate for the critic and value networks.
                Defaults to 0.0003.
            input_dims (List[int], optional): The input dimensions of the environment.
                Defaults to [8].
            hidden_dims (List[int], optional): The hidden dimensions of the networks.
                Defaults to [256, 256].
            n_actions (int, optional): The number of actions or components of that action.
                Defaults to 2.
            batch_size (int, optional): The batch size used for training the networks.
                Defaults to 256.    env.action_space.

            max_size (int, optional): The maximum size of the replay buffer.
                Defaults to 1000000.
            tau (float, optional): The soft update parameter for the target networks.
                Defaults to 0.005.
            env (Optional[None], optional): The environment used for training the agent.
                Defaults to None.
            gamma (float, optional): The discount factor for the rewards.
                Defaults to 0.99.
            reward_scale (int, optional): The scale of the rewards that is dependant on
                the size of the action space. Defaults to 2.
            dtype (torch.dtype, optional): The data type of the tensors.
        """

        # store the input arguments
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.max_size = max_size
        self.tau = tau
        self.env = env
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.dtype = dtype

        self.memory = ReplayBuffer(
            max_size=self.max_size,
            input_shape=self.input_dims,
            n_actions=self.n_actions,
            dtype=self.dtype,
        )

        # create the actor
        self.actor = ActorNetwork(
            alpha=self.alpha,
            input_dims=self.input_dims,
            n_actions=self.n_actions,
            hidden_dims=self.hidden_dims,
            name="actor",
            max_action=self.env.action_space.high,
            dtype=self.dtype,
        )

        # create the critic networks
        self.critic_1 = CriticNetwork(
            beta=self.beta,
            input_dims=self.input_dims,
            n_actions=self.n_actions,
            hidden_dims=self.hidden_dims,
            name="critic_1",
            dtype=self.dtype,
        )
        self.critic_2 = CriticNetwork(
            beta=self.beta,
            input_dims=self.input_dims,
            n_actions=self.n_actions,
            hidden_dims=self.hidden_dims,
            name="critic_2",
            dtype=self.dtype,
        )

        # create the value and target value networks
        self.value = ValueNetwork(
            beta=self.beta,
            input_dims=self.input_dims,
            hidden_dims=self.hidden_dims,
            name="value",
            dtype=self.dtype,
        )
        self.target_value = ValueNetwork(
            beta=self.beta,
            input_dims=self.input_dims,
            hidden_dims=self.hidden_dims,
            name="target_value",
            dtype=self.dtype,
        )

        self.update_network_parameters(tau=1)

    def choose_action(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Choose an action based on the observation.

        Args:
            observation (torch.Tensor): The observation from the environment.

        Returns:
            torch.Tensor: The action to take in the environment.
        """
        action, _ = self.actor.sample_normal(observation, reparameterize=False)
        return action

    def remember(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        new_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """
        Store a transition in the replay buffer.

        Args:
            state (torch.Tensor): The current state of the environment.
            action (torch.Tensor): The action taken in the environment.
            reward (torch.Tensor): The reward received from the environment.
            new_state (torch.Tensor): The new state of the environment.
            done (torch.Tensor): The terminal state of the environment.
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        """
        Update the network parameters for the target value network.

        Args:
            tau (_type_, optional): The soft update parameter for the target networks.
        """
        # if tau is None, set it to the class attribute
        if tau is None:
            tau = self.tau

        # get the parameters for the value and target value networks
        value_params = self.value.named_parameters()
        target_value_params = self.target_value.named_parameters()

        # create dictionaries for the value and target value network parameters
        value_state_dict = dict(value_params)
        target_value_state_dict = dict(target_value_params)

        # update the target value network parameters using the soft update rule.
        # this is done by taking a weighted sum of the value network parameters and
        # the target value network parameters.
        for name in value_state_dict:
            value_state_dict[name] = (
                tau * value_state_dict[name].clone()
                + (1 - tau) * target_value_state_dict[name].clone()
            )

        # load the updated value network parameters into the target value network.
        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        """
        Save the actor, critic and value networks.
        """
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()

    def load_models(self):
        """
        Load the actor, critic and value networks.
        """
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()

    def learn(self):
        """
        Learn from the replay buffer by sampling a batch of transitions and updating the
        value, critic and actor networks.
        """
        # check if the memory is large enough to sample from it and return if it is not.
        if self.memory.mem_counter < self.batch_size:
            return

        # sample a batch of transitions from the replay buffer
        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        # calculate the value and target value for the current and new states.
        value = self.value.forward(state).view(-1)
        target_value = self.target_value.forward(new_state).view(-1)
        # where done is True, set the target value to 0.
        target_value[done] = 0.0

        # sample actions and log probabilities from the actor network for the current
        # state according to the new policy. This is done to calculate the loss for the
        # value and critic networks.
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)

        # calculate the Q values for the current state and actions using the critic
        # networks. The minimum of the two Q values is used to stabilize training and
        # prevent overestimation of the Q values. This is analogous to the double Q
        # learning algorithm using the minimum of the two Q values.
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # calculate the value loss and update the value network.
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)

        # keep the computational graph for the value loss to calculate the actor loss.
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # sample actions and log probabilities from the actor network using the current
        # state and reparameterize the actions.
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)

        # calculate the Q values for the current state and actions using the critic
        # networks taking the minimum of the two Q values.
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # calculate the actor loss and update the actor network.
        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # zero the gradients for the critic networks.
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        # calculate the target Q value for the critic networks.
        q_hat = self.reward_scale * reward + self.gamma * target_value

        # calculate the Q values for the old policy using the current state and actions.
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)

        # calculate the critic loss and update the critic networks.
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()

        # update the critic networks.
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # update the target value network.
        self.update_network_parameters()
