# File Name: Replay_Buffer.py

# File Description: The purpose of this file is to create a replay buffer
# for the project defined below. A replay buffer or memory benefits learning 
# algorithms by allowing the ability to store and recall experience tuples. 

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

import random 
from collections import namedtuple, deque

# ------------------- Create Replay Buffer Class --------------------

class Replay_Buffer:
    """Fixed-size buffer to store experience tuples. """ 
    
    def __init__(self, buffer_size, batch_size): 
        """Initialize a Replay_Buffer object. 
        Parameters
        =======================================
            buffer_size: maximum size of buffer
            batch_size: size of each training batch_size
        """
        # Defining Internal Memory (deque) 
        self.memory = deque(maxlen = buffer_size) 
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        
        self.memory.append(e)
        
    def sample(self, batch_size = 64 ):
        """ Randomly sample a batch of experiences from memory."""
        Batch_Sample = random.sample(self.memory, k = self.batch_size)
        
        return Batch_Sample
        
    def __len__(self): 
        """Return the current size of internal memory."""
        Internal_Memory_Size = len(self.memory)
        
        return Internal_Memory_Size
        