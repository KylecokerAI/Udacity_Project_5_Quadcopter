



import numpy as np
from agents.Agent import DDPG
from agents.Ornstein_Uhlenbeck_Noise import OUNoise 
from Quadcopter_Task import Take_Off as Task

number_of_episodes = 500
Target_Position = np.array([0., 0., 50.])
Take_Off = Task(target_pos = Target_Position)
Agent = DDPG(Take_Off)
Best_Score = -np.inf
Worst_Score = np.inf
Saved_States = np.zeros([18, number_of_episodes])
All_Scores = np.zeros(number_of_episodes)

def Train():
    
    for episode in range(1, number_of_episodes + 1):
        State = Agent.reset_episode()

        Score = 0

        while True: 
            Action = Agent.act(State)

            Next_State, Reward, Done = Take_Off.step(Action)
            Agent.step(Action, Reward, Next_State, Done)
            
            State = Next_State
            Score += Reward

            Best_Score = max(Best_Score, Score)
            Worst_Score = min(Worst_Score, Score)

            if Done: 
                break
                
        Saved_States[:, episode - 1] = State
        All_Scores[episode - 1] = Score

        if Done:
            print("\rEpisode = {:4d}, Score = {:7.3f} (Best = {:7.3f} , Worst = {:7.3f})".format(
               episode, Score, Best_Score, Worst_Score), end="")

            sys.stdout.flush()