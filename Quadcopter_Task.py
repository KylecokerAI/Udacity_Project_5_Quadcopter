import numpy as np
from physics_sim import PhysicsSim

class Take_Off():
    """Take Off defines the goal and provides feedback to the agent."""
    def __init__(self, init_pos = None, init_vel = None, init_omega = None, runtime = 5, target_pos = None):
        
        """Initialize a Task object. 
        Parameters
        =================================================
            init_pos: initial position of the quadcopter in (X, Y, X) dimensions
                      and the Euler angles.
            init_vel: initial velocity of the quadcopter in (X, Y, Z) dimensions
            init_omega: initial angular rate for each Euler Angle (rad/s)
            runtime: time limit for each episode
            target_pos: target (X, Y, Z) position for the agent
            target_vel: target (X, Y, Z) velocitie for the agenet 
        """

        # Simulation Setup
        self.sim = PhysicsSim(init_pos, init_vel, init_omega, runtime)
        self.start_pos = self.sim.pose[:3]
        self.action_repeat = 3

        # Initialize Current State (Coordinate/Angular Position and Velocity) 
        init_pos_size = 3 + 3 # 3 for coordinate position and 3 for angular position
        init_vel_size = 3     # 3 for coordinate velocities 
        init_omega_size = 3   # 3 for angular velocities
        init_total_size = init_pos_size 

        self.state_size = self.action_repeat * init_total_size
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Target Position (i.e. goal for agent to achieve)
        self.target_pos = target_pos
        self.Margin = 0.3 # 10% Margin for error 
                
    def get_reward(self):
        Penatly_Ratio = 0.003
        dt = 1/50 # from PhysicsSim.py
       
        Position_Error = np.linalg.norm((self.sim.pose[:3] - self.target_pos), axis = 0)
        Error_Percentage = Position_Error/100
        
        if Error_Percentage <= self.Margin: 
            Penalty = -Penatly_Ratio * Position_Error
        else: 
            Penalty =  Penatly_Ratio * Position_Error
            
        # Penalty += 0.5 * Penatly_Ratio * abs(self.sim.linear_accel[2]) * dt ** 2

        Reward = np.tanh((1 - Penalty))
        
        return Reward


    def step(self, action):
        """ Use action to obtain next state, reward, and done. """
        Reward = 0 
        Pos_All = []

        for _ in range(self.action_repeat):
            Done = self.sim.next_timestep(action) # update the sim position and velocity
            Reward += self.get_reward()
            Pos_All.append(self.sim.pose)
            
            # We need to reward the agent for achieving the desired target
            if (self.sim.pose[2] >= self.start_pos[2]) and (self.sim.pose[2] <= self.target_pos[2]):
                Reward += 0.1 
            
            # If the agent produces a negative results; otherwise, it should be penalized. 
            if Done is True and Reward > 0:
                Reward += 0.01
            elif Done is True and Reward <= 0: 
                Reward -= 0.01
                
                
                        
            
        Next_State = np.concatenate(Pos_All)

        return Next_State, Reward, Done              


    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        State = np.concatenate([self.sim.pose] * self.action_repeat)

        return State
            
                
                