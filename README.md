# Safe Policy Learning in Stochastic Constrained Environments

This project investigates policy optimization in environments where safety constraints must be respected during learning. We implement and compare two methods:
- **REINFORCE with constraint penalties**
- **Direct Preference Optimization (DPO)** using trajectory comparisons

We evaluate their effectiveness on a simple 2D point-mass maze navigation task with rectangular obstacles and stochastic dynamics.

The way that the repo is set up is as follows:
- /src contains all the main files
- /code has testing code with many plots

environment.py has environment step requirements
policy.py has the exact policy for everything
train.py has all the training code
utils.py has some useful plotting code

For most of my testing, I used test.ipynb, you have to have the following libraries installed as well:
- Numpy
- Pandas
- Matplotlib

To train on REINFORCE, you can just use train_policy(), and for DPO, you can just run train_DPO_policy() from train.py
