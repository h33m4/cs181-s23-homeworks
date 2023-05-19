import numpy as np
import numpy.random as npr
import pygame as pg

X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900


class Learner(object):
    """
    This agent jumps randomly.
    """

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

        # Initialize our hyperparameters
        self.alpha = 0.1
        self.gamma = 0.1

        self.min_epsilon = 0.001
        self.initial_epsilon = 0.01
        self.epsilon = self.initial_epsilon
        self.eps_decay_rate = 0.1

        self.total_moves = 0


        # We initialize our Q-value grid that has an entry for each action and state.
        # (action, rel_x, rel_y)
        self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def discretize_state(self, state):
        """
        Discretize the position space to produce binned features.
        rel_x = the binned relative horizontal distance between the monkey and the tree
        rel_y = the binned relative vertical distance between the monkey and the tree
        """

        rel_x = int((state["tree"]["dist"]) // X_BINSIZE)
        rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)
        return (rel_x, rel_y)

    def decay_epsilon(self):
        """
        Decay epsilon based on total number of moves, initial epsilon and current epsilon
        """
        self.epsilon = self.min_epsilon + (self.initial_epsilon - self.min_epsilon) * np.exp(-self.eps_decay_rate * self.total_moves)

    def action_callback(self, state):
        """
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        """

        # Discretize the state
        current_state = self.discretize_state(state) # (rel_x, rel_y)

        # Update Q
        if self.last_state is not None:
            # Get the Q values for the last state
            last_state = self.last_state
            last_action = self.last_action
            last_reward = self.last_reward

            # Get the Q value for the last state and action
            last_Q = self.Q[last_action][last_state]

            # Get the Q value for the best action in the current state
            current_Q = np.max(self.Q[:, current_state[0], current_state[1]])

            # Update the Q value for the last state and action
            self.Q[last_action][last_state] = last_Q + self.alpha * (last_reward + self.gamma * current_Q - last_Q)

        # Decay epsilon
        self.total_moves += 1
        self.decay_epsilon()

        # Choose the next action using an epsilon-greedy policy
        if npr.rand() < self.epsilon:
            # Choose a random action
            new_action = npr.randint(2)

        else:
            # Choose the best action
            new_action = np.argmax(self.Q[:, current_state[0], current_state[1]])

        new_state = current_state

        self.last_action = new_action
        self.last_state = new_state

        return self.last_action

    def reward_callback(self, reward):
        """This gets called so you can see what reward you get."""

        self.last_reward = reward
