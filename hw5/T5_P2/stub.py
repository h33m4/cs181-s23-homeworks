# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

# uncomment this for animation
#from SwingyMonkey import SwingyMonkey

# uncomment this for no animation
from SwingyMonkeyNoAnimation import SwingyMonkey

from Learner import Learner

def run_games(learner, hist, iters=100, t_len=100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.
    """
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


def total_score(hist):
    return np.sum(hist)

'''
Grid Search Function to find the best hyperparameters
Searches over alpha, gamma, epsilon, epsilon decay rate
Evaluates the total score over 100 games
'''
def grid_search():
    '''
    Grid Search Function to find the best hyperparameters
    Searches over alpha, gamma, epsilon, epsilon decay rate
    Evaluates the total score over 100 epochs in a game
    Could be improved to have several runs 
    '''
    params = {
        'alpha': [0.01, 0.1, 0.5, 0.9],
        'gamma': [0.01, 0.1, 0.5, 0.9],
        'initial_epsilon': [0.01, 0.1, 0.5],
        'eps_decay_rate': [0.01, 0.1, 0.5]
    }

    # Initialize the best score to be 0
    best_score = 0

    # Initialize the best hyperparameters to be None
    best_params = None

    # Iterate over all hyperparameter configurations
    for alpha in params['alpha']:
        for gamma in params['gamma']:
            for initial_epsilon in params['initial_epsilon']:
                for eps_decay_rate in params['eps_decay_rate']:
                    # Initialize the learner
                    learner = Learner(alpha=alpha, gamma=gamma, initial_epsilon=initial_epsilon, eps_decay_rate=eps_decay_rate)

                    # Initialize the history
                    hist = []

                    # Run games
                    run_games(learner, hist, 100, 100)

                    # Get the total score
                    score = total_score(hist)

                    # Update the best score and best hyperparameters
                    if score > best_score:
                        best_score = score
                        best_params = (alpha, gamma, initial_epsilon, eps_decay_rate)

    return best_score, best_params

def plot_scores():
    '''
    Function to plot the monkey's score over time for different
    hyperparameter configurations
    '''
    import matplotlib.pyplot as plt
    # Initialize the hyperparameter space
    params = {
        'alpha': [0.01, 0.1, 0.01, 0.9],
        'gamma': [0.01, 0.1, 0.5, 0.9],
        'initial_epsilon': [0.001, 0.001, 0.01, 0.01],
        'eps_decay_rate': [0.01, 0.1, 0.5, 0.5]
    }
    for i in range(4):
        alpha = params['alpha'][i]
        gamma = params['gamma'][i]
        initial_epsilon = params['initial_epsilon'][i]
        eps_decay_rate = params['eps_decay_rate'][i]
        # Initialize the learner
        learner = Learner(alpha=alpha, gamma=gamma, initial_epsilon=initial_epsilon, eps_decay_rate=eps_decay_rate)

        # Initialize the history
        hist = []

        # Run games
        run_games(learner, hist, 100, 100)

        # Plot the scores
        plt.plot(hist, label="a: {}, g: {}, e_init: {}, e_decay: {}".format(alpha, gamma, initial_epsilon, eps_decay_rate))
    
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Score vs. Epoch')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. You can update t_len to be smaller to run it faster.
    run_games(agent, hist, 100, 100)
    print(hist)

    plot_scores()
    print(grid_search())

    # Save history. 
    np.save('hist', np.array(hist))
