import numpy as np
from environment import step_environment
from utils import plot_trajectory

def rollout(policy, initial_state, target, horizon=100):
    """
    Simulate one trajectory rollout.
    
    Args:
        policy: policy object
        initial_state: np.array, shape (2,)
        target: np.array, shape (2,)
        horizon: number of steps per episode
        
    Returns:
        states: array of states over trajectory
        actions: array of actions over trajectory
        rewards: array of rewards over trajectory
    """
    state = initial_state
    states, actions, rewards = [], [], []
    
    for _ in range(horizon):
        action, _ = policy.act(state)
        next_state = step_environment(state, action)
        
        distance = np.linalg.norm(next_state - target)
        reward = -distance  # negative distance

        if distance < 0.1:
            reward += 10  # bonus for reaching close to target
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state

    return np.array(states), np.array(actions), np.array(rewards)

def compute_returns(rewards, gamma=0.99):
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

    for episode in range(num_episodes):
        initial_state = np.random.randn(2)  # Start from random position
        states, actions, rewards = rollout(policy, initial_state, target, horizon)
        
        returns = compute_returns(rewards)

        # Normalize advantages
        advantages = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Policy update
        policy.update(states, actions, advantages)

        total_reward = np.sum(rewards)
        reward_history.append(total_reward)

        # Optional: plot trajectory every N episodes
        if episode % 50 == 0:
            plot_trajectory(states, target)
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

    return reward_history