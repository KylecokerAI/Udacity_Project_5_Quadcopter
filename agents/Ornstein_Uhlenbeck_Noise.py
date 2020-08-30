# File Name: Ornstein_Uhlenbeck_Noise.py 

# File Description: We'll use a specific noise process that has some 
#   desired properties, called the Ornstein–Uhlenbeck process. 
#   It essentially generates random samples from a Gaussian (Normal) distribution,
#   but each sample affects the next one such that two consecutive samples are 
#   more likely to be closer together than further apart. In this sense, 
#   the process in Markovian in nature

# Used in the following Project: 
#       Project Name: Teach a Quadcopter How to Fly

# References: 
#   1.) Udacity - Deep Learning Course 
#   2.) Lillicrap, Timothy P., et al., 2015. Continuous Control with Deep Reinforcement Learning.
#       Link: https://arxiv.org/pdf/1509.02971.pdf 

# Created By: Kyle Coker
# Creation Date: August 23rd 2020
# Modfied Date: ------

# --------------------- Initial Setup -------------------------------

import numpy as np
import copy

# ------------------- Create Noise Class ---------------------------

class OUNoise:
    """Ornstein-Uhlenbeck process."""
    
    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma 
        self.reset()
        
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        
        return self.state 