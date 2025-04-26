import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

def plot_trajectory(states, target):
    """
    Plot a 2D trajectory towards a target.
    
    Args:
        states: np.array of shape (T, 2)
        target: np.array of shape (2,)
    """
    plt.figure(figsize=(6,6))
    plt.plot(states[:,0], states[:,1], marker='o', label='Trajectory')
    plt.scatter(states[0,0], states[0,1], color='green', s=100, label='Start')
    plt.scatter(target[0], target[1], color='red', s=100, label='Target')
    plt.title('Agent Trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_rewards(rewards):
    plt.figure(figsize=(8,5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.grid(True)
    plt.show()
