import numpy as np

class StochasticPolicyNetwork:
    def __init__(self, state_dim, action_dim, hidden_dim=16, noise_std=0.1):
        self.W1 = np.random.randn(hidden_dim, state_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(action_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(action_dim)
        self.noise_std = noise_std
        self.lr = 0.0003

    def act(self, state):
        h = np.maximum(0, self.W1 @ state + self.b1)  # ReLU
        mean_action = self.W2 @ h + self.b2
        action = mean_action + np.random.normal(0, self.noise_std, size=mean_action.shape)
        return action, mean_action

    def update(self, states, actions, advantages, weight=1.0):
        for state, action, advantage in zip(states, actions, advantages):
            h = np.maximum(0, self.W1 @ state + self.b1)  # ReLU
            mean_action = self.W2 @ h + self.b2
            grad_logp = (action - mean_action) / (self.noise_std ** 2)
            grad_logp = np.clip(grad_logp, -5.0, 5.0)

            # Gradients for W2, b2
            grad_W2 = np.outer(grad_logp, h)
            grad_b2 = grad_logp

            # Gradients for W1, b1 (through ReLU)
            dh = (self.W2.T @ grad_logp) * (h > 0)
            grad_W1 = np.outer(dh, state)
            grad_b1 = dh

            # Update
            self.W2 += self.lr * advantage * grad_W2
            self.b2 += self.lr * advantage * grad_b2
            self.W1 += self.lr * advantage * grad_W1
            self.b1 += self.lr * advantage * grad_b1

        # Regularize
        self.W1 *= 0.99
        self.W2 *= 0.99
    
    def updateDPO(self, states, actions, advantages, weight=1.0):
        for state, action, advantage in zip(states, actions, advantages):
            h = np.maximum(0, self.W1 @ state + self.b1)  # ReLU
            mean_action = self.W2 @ h + self.b2
            grad_logp = (action - mean_action) / (self.noise_std ** 2)
            grad_logp = np.clip(grad_logp, -5.0, 5.0)

            # Gradients for W2, b2
            grad_W2 = np.outer(grad_logp, h)
            grad_b2 = grad_logp

            # Gradients for W1, b1 (through ReLU)
            dh = (self.W2.T @ grad_logp) * (h > 0)
            grad_W1 = np.outer(dh, state)
            grad_b1 = dh

            # Apply update with DPO weight scaling
            scale = self.lr * advantage * weight
            self.W2 += scale * grad_W2
            self.b2 += scale * grad_b2
            self.W1 += scale * grad_W1
            self.b1 += scale * grad_b1

        # Regularization
        self.W1 *= 0.99
        self.W2 *= 0.99
