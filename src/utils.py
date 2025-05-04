import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

import matplotlib.pyplot as plt

def plot_trajectory(traj, start, target, title="Best Episode Trajectory",
                    walls=None, save_path=None):

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
    # Draw rectangular walls
    if walls is not None:
        for (x, y, w, h) in walls:
            rect = plt.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor='black', facecolor='gray', alpha=0.5
            )
            plt.gca().add_patch(rect)


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


def compare_dpo_vs_reinforce(reinforce_returns, dpo_returns,
                              reinforce_violations, dpo_violations,
                              smooth_window=5):

    def safe_smooth(x, k):
        if len(x) < k or k <= 0:
            print(f"[Warning] Not enough data to smooth (len={len(x)}, window={k})")
            return np.array([])
        return np.convolve(x, np.ones(k)/k, mode='valid')

    if len(reinforce_returns) == 0:
        print("[Error] `reinforce_returns` is empty.")
        return
    if len(dpo_returns) == 0:
        print("[Error] `dpo_returns` is empty.")
        return
    if len(reinforce_violations) == 0:
        print("[Error] `reinforce_violations` is empty.")
        return
    if len(dpo_violations) == 0:
        print("[Error] `dpo_violations` is empty.")
        return

    smoothed_reinforce = safe_smooth(reinforce_returns, smooth_window)
    smoothed_dpo = safe_smooth(dpo_returns, smooth_window)
    smoothed_violations_reinforce = safe_smooth(reinforce_violations, smooth_window)
    smoothed_violations_dpo = safe_smooth(dpo_violations, smooth_window)

    min_len = min(len(smoothed_reinforce), len(smoothed_dpo),
                  len(smoothed_violations_reinforce), len(smoothed_violations_dpo))

    if min_len == 0:
        print("[Error] Smoothed data is empty, skipping plot.")
        return

    episodes = np.arange(min_len)

    plt.figure(figsize=(12, 5))

    # Return comparison
    plt.subplot(1, 2, 1)
    plt.plot(episodes, smoothed_reinforce[:min_len], label='REINFORCE', linewidth=2)
    plt.plot(episodes, smoothed_dpo[:min_len], label='DPO', linewidth=2)
    plt.title("Smoothed Return Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("smoothed_return_comparison.png", dpi=300)

    # Violation comparison
    plt.subplot(1, 2, 2)
    plt.plot(episodes, smoothed_violations_reinforce[:min_len], label='REINFORCE', linewidth=2)
    plt.plot(episodes, smoothed_violations_dpo[:min_len], label='DPO', linewidth=2)
    plt.title("Smoothed Violations Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Violations")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig("smoothed_violations_comparison.png", dpi=300)