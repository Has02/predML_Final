import numpy as np
from environment import step_environment
from utils import plot_trajectory

def violates_constraint(state, center, radius):
    return np.linalg.norm(state - center) < radius

def rollout_episode(policy, initial_state, target, horizon=50,
                    obstacle_center=np.array([0.5, 0.5]),
                    obstacle_radius=0.2,
                    constraint_penalty=-5.0):
    state = initial_state
    states, actions, rewards = [], [], []

    violations = 0

    for _ in range(horizon):
        action, _ = policy.act(state)
        next_state = step_environment(state, action)

        # Reward is negative distance to target (as before)
        reward = -np.linalg.norm(next_state - target) ** 2

        # Apply constraint penalty
        if violates_constraint(next_state, obstacle_center, obstacle_radius):
            reward += constraint_penalty
            violations += 1

        # Store
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    return np.array(states), np.array(actions), np.array(rewards), state, violations





def compute_reward_to_go(rewards, gamma=0.99):
    """
    Compute discounted returns (reward-to-go).
    """
    returns = np.zeros_like(rewards)
    running_sum = 0
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        returns[t] = running_sum
    return returns

def train_policy(policy, num_episodes=1000, horizon=50):
    """
    Train the policy using REINFORCE.
    """
    target = np.array([0.0, 0.0])  # Target location
    reward_history = []

    episode_returns = []
    final_dists = []

    best_dist = float('inf')
    best_trajectory = None

    for episode in range(num_episodes):
        initial_state = np.random.randn(2)
        states, actions, rewards, final_state, violations = rollout_episode(policy, initial_state, target, horizon)
        
        advantages = compute_reward_to_go(rewards)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        policy.update(states, actions, advantages)

        total_reward = np.sum(rewards)
        final_dist = np.linalg.norm(final_state - target)

        episode_returns.append(total_reward)
        final_dists.append(final_dist)

        if final_dist < best_dist:
            best_dist = final_dist
            best_trajectory = (states, actions)

        if episode % 10 == 0:
            print(f"Episode {episode}: Total Reward {total_reward:.2f}, Final Dist {final_dist:.2f}, Violations: {violations}")

    return episode_returns, final_dists, best_trajectory
