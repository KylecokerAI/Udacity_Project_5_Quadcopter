# File Name: Actor.py

# File Description: This file is used to create the Actor class 
#   for the project defined below. The actor class is being used 
#   in the evaluation of a Deep Deterministic Policy Gradient (DDPG). 
#   The Actor class will be coupled with a Critic class. In the case
#   of the actor class its purpose is to take the input of a state
#   and outputs the best action. It essentially controls how the 
#   Agent class behaves by learning the optimal policy. 

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

from keras import layers, models, optimizers
from keras import backend as K 
import numpy as np

# ------------------- Create Actor Class ----------------------------

class Actor: 
    """Actor (Policy) Model."""
    
    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model. 
        
        Parameters 
        ===============================================================
            state_size (init): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Minimum value of each action dimension 
            action_high (array): Maximum value of each action dimension
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        
        self.build_model()
        
    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        
        Regularizer = layers.regularizers.l2(1e-6)
        
        # Define input layer (states) 
        
        states = layers.Input(shape = (self.state_size, ), name = 'states')
        
        # Add hidden layers 
        Hidden_Layer_1 = layers.Dense(units = 400, kernel_regularizer = Regularizer)(states)
        Batch_Norm_1 = layers.BatchNormalization()(Hidden_Layer_1)
        Activate_1 = layers.Activation("relu")(Batch_Norm_1)
        
        Hidden_Layer_2 = layers.Dense(units = 300, kernel_regularizer = Regularizer)(Activate_1)
        Batch_Norm_2 = layers.BatchNormalization()(Hidden_Layer_2)
        Activate_2 = layers.Activation("relu")(Batch_Norm_2)
        
        #Hidden_Layer_3 = layers.Dense(units = 200, kernel_regularizer = Regularizer)(Activate_2)
        #Batch_Norm_3 = layers.BatchNormalization()(Hidden_Layer_3)
        #Activate_3 = layers.Activation("relu")(Batch_Norm_3)
        
        
        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units = self.action_size, 
                                   activation = 'sigmoid', 
                                   kernel_initializer = layers.initializers.RandomUniform(minval = -0.003, maxval = 0.003),
                                   name = 'raw_actions')(Activate_2)
                                   
        # Scale [0, 1] output for each action dimension to proper range. 
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, 
                                name = 'actions')(raw_actions)
        
        # Create Keras model
        self.model = models.Model(inputs = states, outputs = actions)
        
        # Define loss function using action value (Q value) gradients. 
        # Note: These gradients will need to be computed using the critic
        #   model, and fed in while training. Hence it is specified as 
        #   part of the "inputs" used in the training function below.
        action_gradients = layers.Input(shape = (self.action_size,))
        loss = K.mean(-action_gradients * actions)
        
        # Define optimizer and training function 
        optimizer = optimizers.Adam(lr = 3e-4)
        updates_op = optimizer.get_updates(params = self.model.trainable_weights, 
                                           loss = loss)
        self.train_fn = K.function(
                            inputs = [self.model.input, action_gradients, K.learning_phase()],
                            outputs = [],
                            updates = updates_op)
                            
        
        
        
        