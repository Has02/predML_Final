import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

def plot_trajectory(traj, start, target, title="Best Episode Trajectory",
                    obstacle_center=None, obstacle_radius=None, save_path=None):
    """
    Plots the trajectory of an agent given a sequence of states.

    Parameters:
        traj (ndarray): Array of shape (T, 2) with agent positions over time
        start (ndarray): Initial position (2D)
        target (ndarray): Target position (2D)
        obstacle_center (ndarray): Center of the circular constraint region
        obstacle_radius (float): Radius of the constraint region
        title (str): Plot title
        save_path (str): Optional path to save the figure
    """
    plt.figure(figsize=(6, 6))
    plt.plot(traj[:, 0], traj[:, 1], marker='o', markersize=2, label='Trajectory')
    plt.scatter(start[0], start[1], color='green', s=100, label='Start')
    plt.scatter(target[0], target[1], color='red', s=100, label='Target')

    # Draw constraint violation region
    if obstacle_center is not None and obstacle_radius is not None:
        circle = Circle(obstacle_center, obstacle_radius, color='gray', alpha=0.3, label='Constraint Region')
        plt.gca().add_patch(circle)

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()



def plot_rewards(rewards):
    plt.figure(figsize=(8,5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.grid(True)
    plt.show()
