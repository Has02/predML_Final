# src/environment.py

import numpy as np

def step_environment(state, action, noise_std=0.05):
    """
    Simple 2D point mass dynamics with Gaussian noise.
    
    Args:
        state: np.array of shape (2,)
        action: np.array of shape (2,)
        noise_std: standard deviation of noise
        
    Returns:
        next_state: np.array of shape (2,)
    """
    next_state = state + 0.1 * action + np.random.normal(0, noise_std, size=state.shape)
    return next_state
