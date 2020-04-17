"""Linear Q-learning agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging

import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools


class LinearQLearner(rl_agent.AbstractAgent):
    """Linear Q-Learning agent.
    """

    def __init__(self,
                 player_id,
                 num_actions,
                 state_representation_size,
                 step_size=0.5,
                 epsilon_schedule=rl_tools.ConstantSchedule(0.2),
                 discount_factor=1.0,
                 initialization='random',
                 name='linear_qlearner'):
        """Initialize the Q-Learning agent."""
        self._player_id = player_id
        self._num_actions = num_actions
        self._state_representation_size = state_representation_size
        self._step_size = step_size
        self._epsilon_schedule = epsilon_schedule
        self._epsilon = epsilon_schedule.value
        self._discount_factor = discount_factor
        # random init state features + one hot encoded actions for (S,a)
        if initialization == 'random':
            self._weights = np.random.random(self._state_representation_size * self._num_actions)
        elif initialization == 'zero':
            self._weights = np.zeros(self._state_representation_size * self._num_actions)
        elif type(initialization) is list and len(initialization) == self._state_representation_size * self._num_actions:
            self._weights = initialization
        else:
            raise ValueError("Not implemented, choose initialization from 'random', 'zero' and a custom list.")
        self._prev_info_state = None

    def q_value(self, info_state, action):
        action_weights = self._weights[
                         self._state_representation_size * action:self._state_representation_size * (action + 1)]
        return np.dot(action_weights, info_state)

    def _epsilon_greedy(self, info_state, legal_actions, epsilon):
        """Returns a valid epsilon-greedy action and valid action probs.

        If the agent has not been to `info_state`, a valid random action is chosen.

        Args:
          info_state: hashable representation of the information state.
          legal_actions: list of actions at `info_state`.
          epsilon: float, prob of taking an exploratory action.

        Returns:
          A valid epsilon-greedy action and valid action probabilities.
        """
        probs = np.zeros(self._num_actions)
        greedy_q = max([self.q_value(info_state, a) for a in legal_actions])
        greedy_actions = [
            a for a in legal_actions if self.q_value(info_state, a) == greedy_q
        ]
        probs[legal_actions] = epsilon / len(legal_actions)
        probs[greedy_actions] += (1 - epsilon) / len(greedy_actions)
        action = np.random.choice(range(self._num_actions), p=probs)
        return action, probs

    def step(self, time_step, is_evaluation=False):
        """Returns the action to be taken and updates the Q-values if needed.

        Args:
          time_step: an instance of rl_environment.TimeStep.
          is_evaluation: bool, whether this is a training or evaluation call.

        Returns:
          A `rl_agent.StepOutput` containing the action probs and chosen action.
        """
        info_state = time_step.observations["info_state"][self._player_id]
        legal_actions = time_step.observations["legal_actions"][self._player_id]

        # Prevent undefined errors if this agent never plays until terminal step
        action, probs = None, None

        # Act step: don't act at terminal states.
        if not time_step.last():
            epsilon = 0.0 if is_evaluation else self._epsilon
            action, probs = self._epsilon_greedy(
                info_state, legal_actions, epsilon=epsilon)

        # Learn step: don't learn during evaluation or at first agent steps.
        if self._prev_info_state and not is_evaluation:
            target = time_step.rewards[self._player_id]
            if not time_step.last():  # Q values are zero for terminal.
                target += self._discount_factor * max(
                    [self.q_value(info_state, a) for a in legal_actions])

            prev_q_value = self.q_value(self._prev_info_state, self._prev_action)
            self._last_loss_value = target - prev_q_value

            # update weights that correspond to (the previous) action that led us to the reward target
            self._weights[self._state_representation_size * self._prev_action:self._state_representation_size * (
                    self._prev_action + 1)] += self._step_size * self._last_loss_value * np.array(self._prev_info_state)
            # self._q_values[self._prev_info_state][self._prev_action] += (
            #     self._step_size * self._last_loss_value)

            # Decay epsilon, if necessary.
            self._epsilon = self._epsilon_schedule.step()

            if time_step.last():  # prepare for the next episode.
                self._prev_info_state = None
                return

        # Don't mess up with the state during evaluation.
        if not is_evaluation:
            self._prev_info_state = info_state
            self._prev_action = action
        return rl_agent.StepOutput(action=action, probs=probs)

    def take_action(self, time_step, action, target):
        """Returns the action to be taken and updates the Q-values using Monte-Carlo update.

        Args:
          time_step: an instance of rl_environment.TimeStep.
          action: int, an action to take.
          target: float, the reward in the end of the episode.
        Returns:
          None
        """
        info_state = time_step.observations["info_state"][self._player_id]

        if self._prev_info_state:
            prev_q_value = self.q_value(self._prev_info_state, self._prev_action)
            self._last_loss_value = target - prev_q_value
            # update weights that correspond to (the previous) action that led us to the target in the end of episode
            self._weights[self._state_representation_size * self._prev_action:self._state_representation_size * (
                    self._prev_action + 1)] += self._step_size * self._last_loss_value * np.array(self._prev_info_state)

            if time_step.last():  # prepare for the next episode.
                self._prev_info_state = None
                return

        self._prev_info_state = info_state
        self._prev_action = action

        return None

    def save(self, checkpoint_dir):
        # trained_agents / id. generator_name / weights
        os.mkdir(checkpoint_dir)
        path = checkpoint_dir + '/' + str(self._player_id) + ' weights.npy'
        np.save(path, self._weights)
        logging.info("saved to path: %s", path)

    def restore(self, checkpoint_dir):
        path = checkpoint_dir + '/' + str(self._player_id) + ' weights.npy'
        logging.info("Restoring checkpoint: %s", (path))
        self._weights = np.load(path)

    @property
    def loss(self):
        return self._last_loss_value
