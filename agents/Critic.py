# File Name: Critic.py

# File Description: This file is used to create the Critic class 
#   for the project defined below. The critic class is being used 
#   in the evaluation of a Deep Deterministic Policy Gradient (DDPG). 
#   The Critic class will be coupled with a Actor class. In the case
#   of the critic class its purpose is to evaluate the action produced
#   by the actor class by computing a value function. 

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

# ------------------- Create Critic Class ---------------------------

class Critic:
    """Critic (Value) Model."""
    
    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        self.build_model()
        
    def build_model(self):
        """Build a critic (value) network that maps (state, action) 
            pairs -> Q-Values."""
        
        Regularizer = layers.regularizers.l2(1e-6)
        
        # Define input layers
        states = layers.Input(shape = (self.state_size, ), name = 'states')
        actions = layers.Input(shape = (self.action_size, ), name = 'actions')
        
        # Add hidden layer(s) for state pathway
        Net_State_1 = layers.Dense(units = 400, kernel_regularizer = Regularizer)(states)
        Batch_Norm_1 = layers.BatchNormalization()(Net_State_1)
        Activate_1 = layers.Activation("relu")(Batch_Norm_1)
        
        Net_State_2 = layers.Dense(units = 300, kernel_regularizer = Regularizer)(Activate_1)
        #Batch_Norm_2 = layers.BatchNormalization()(Net_State_2)
        #Activate_2 = layers.Activation("relu")(Batch_Norm_2)
        
        Net_States = Net_State_2
        
        # Add hidden layer(s) for action pathway
        Net_Actions = layers.Dense(units = 300, kernel_regularizer = Regularizer)(actions)
        #Net_Actions = layers.BatchNormalization()(Net_Actions)
        #Net_Actions = layers.Activation("relu")(Net_Actions)
        
        
        # Combine state and action pathways
        Net_Layer_Combo = layers.add([Net_States, Net_Actions])
        #Net_Output = layers.Dense(units = 200, kernel_regularizer = Regularizer)(Net_Layer_Combo)
        #Net_Output = layers.BatchNormalization()(Net_Output)
        Net_Output = layers.Activation("relu")(Net_Layer_Combo)
        
        # Add final output layer to produce action values (Q Values)
        Q_Values = layers.Dense(units = 1, 
                                kernel_initializer = layers.initializers.RandomUniform(minval = -0.003, maxval = 0.003),
                                name = 'Q_Values')(Net_Output)
        
        # Create Keras model
        self.model = models.Model(inputs = [states, actions], outputs = Q_Values)
        
        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr = 0.01)
        self.model.compile(optimizer = optimizer, loss = 'mse')
        
        # Compute action gradients (derivative of Q values w.r.t. actions) 
        action_gradients = K.gradients(Q_Values, actions)
        
        # Define an additional function to fetch action gradients 
        #   (to be used by the actor model).
        self.get_action_gradients = K.function(
                                    inputs = [*self.model.input, K.learning_phase()],
                                    outputs = action_gradients)
                                    
            
        
        
        
        
        
        
        
        
        
        
        
        