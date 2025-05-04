import numpy as np
from environment import step_environment
from utils import plot_trajectory

def violates_maze(state):
    x, y = state

    # Vertical corridor: x in [0.05, 0.25], y in [0.05, 0.95]
    in_vertical = 0.05 <= x <= 0.25 and 0.05 <= y <= 0.95

    # Horizontal corridor: x in [0.25, 0.95], y in [0.75, 0.95]
    in_horizontal = 0.25 <= x <= 0.95 and 0.75 <= y <= 0.95

    return not (in_vertical or in_horizontal)

def rollout_episode(policy, initial_state, target, horizon=50,
                    walls=None,
                    constraint_penalty=-5.0):
    state = initial_state
    states, actions, rewards = [], [], []
    violations = 0

    for _ in range(horizon):
        action, _ = policy.act(state)
        next_state = step_environment(state, action)

        reward = -np.linalg.norm(next_state - target) ** 2

        if violates_maze(next_state):
            reward += constraint_penalty
            violations += 1

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    return np.array(states), np.array(actions), np.array(rewards), state, violations, walls

def compute_reward_to_go(rewards, gamma=0.99):
    # reward = -||next_state - target||^2 + λ * constraint_penalty * I[violation]
    returns = np.zeros_like(rewards)
    running_sum = 0
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        returns[t] = running_sum
    return returns

def train_policy(policy, num_episodes=1000, horizon=50,
                 constraint_penalty=-5.0,
                 log_violations=True):
    initial_state = np.array([0.1, 0.1])     # bottom-left corner
    target = np.array([0.9, 0.9])    # top-right corner

    episode_returns = []
    final_dists = []
    violation_counts = []

    best_dist = float('inf')
    best_trajectory = None
    best_walls = None

    # Each wall is defined as: [x_start, y_start, width, height]
    walls = [
        # Left vertical outer wall
        (0.0, 0.0, 0.05, 1.0),

        # Right wall of vertical corridor
        (0.25, 0.0, 0.05, 0.75),

        # Bottom wall (fully enclosed)
        (0.0, 0.0, 0.3, 0.05),

        # Top wall of horizontal corridor
        (0, 0.95, 1.0, 0.05),

        # Bottom wall of horizontal corridor
        (0.25, 0.7, 0.75, 0.05),

        # Far right vertical wall (close right corridor)
        (0.95, 0.75, 0.05, 0.25),
    ]



    for episode in range(num_episodes):
        states, actions, rewards, final_state, violations, walls_used = rollout_episode(
            policy,
            initial_state,
            target,
            horizon,
            walls=walls,
            constraint_penalty=constraint_penalty
        )

        if log_violations:
            violation_counts.append(violations)

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
            best_walls = walls

        if episode % 10 == 0:
            print(f"Episode {episode}: Total Reward {total_reward:.2f}, Final Dist {final_dist:.2f}, Violations: {violations}")

    if not log_violations:
        violation_counts = []
    if log_violations:
        print(f"Start: {initial_state}, Target: {target}")


    return episode_returns, final_dists, best_trajectory, violation_counts, best_walls


def dpo_weight_stable(R_pref, R_other, beta=1.0):
    max_r = max(beta * R_pref, beta * R_other)
    numerator = np.exp(beta * R_pref - max_r)
    denominator = np.exp(beta * R_pref - max_r) + np.exp(beta * R_other - max_r)
    return numerator / denominator

def train_dpo_policy(policy, num_epochs=100, episodes_per_epoch=10, beta=0.01, 
                     max_timesteps=50, constraint_penalty=-5.0, noise_std=0.05):
    all_returns = []
    all_dists = []
    all_violations = []

    for epoch in range(num_epochs):
        trajectories = []

        for _ in range(episodes_per_epoch):
            state = np.array([0.2, 0.1])
            target = np.array([0.8, 0.9])
            states, actions, rewards = [], [], []
            violations = 0

            for _ in range(max_timesteps):
                action, _ = policy.act(state)
                next_state = step_environment(state, action, noise_std=noise_std)

                violated = violates_maze(next_state)
                reward = -np.linalg.norm(next_state - target) ** 2
                if violated:
                    reward += constraint_penalty
                    violations += 1

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state

            total_return = np.clip(sum(rewards), -1e3, 1e3)
            dist_to_target = np.linalg.norm(state - target)

            trajectories.append({
                'states': np.array(states),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'return': total_return,
                'distance': dist_to_target,
                'violations': violations
            })

        # Preference pairs: (A preferred over B if higher return)
        for i in range(len(trajectories)):
            for j in range(i+1, len(trajectories)):
                traj_A = trajectories[i]
                traj_B = trajectories[j]

                R_A = np.clip(traj_A['return'], -1e3, 1e3)
                R_B = np.clip(traj_B['return'], -1e3, 1e3)
                mean_R = np.mean([R_A, R_B])
                std_R = np.std([R_A, R_B]) + 1e-8
                R_A = (R_A - mean_R) / std_R
                R_B = (R_B - mean_R) / std_R


                logits = np.array([R_A, R_B]) * beta
                logits -= np.max(logits)  # softmax stability
                weights = np.exp(logits)
                weight = weights[0] / (weights[0] + weights[1] + 1e-8)  # avoid divide by 0

                preferred = traj_A if R_A > R_B else traj_B
                states = preferred['states']
                actions = preferred['actions']
                returns = compute_reward_to_go(preferred['rewards'])

                # Normalize returns for better stability
                returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

                policy.updateDPO(states, actions, returns, weight=weight)

        best_traj = max(trajectories, key=lambda t: t['return'])
        all_returns.append(best_traj['return'])
        all_dists.append(best_traj['distance'])
        all_violations.append(best_traj['violations'])

        print(f"[DPO] Epoch {epoch} | Return: {best_traj['return']:.2f} | Violations: {best_traj['violations']}")

    return all_returns, all_dists, all_violations
