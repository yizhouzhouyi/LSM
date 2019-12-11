# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 09:55:20 2018

@author: yizhou
"""

from environment import Maze
from q_learning import QLearningTable

def update():          
    total_reward_value = 0
    average_reward = []
    for episode in range(2000):
        observation = env.reset()
        #total_reward_value = 0
        while True:
            env.render()
# =============================================================================
#             if (episode+1) >= 3:
#                 action = RL.choose_ma_action(str(observation))
#             else:
#                 action = RL.choose_action(str(observation))
# =============================================================================
            action = RL.choose_ma_action(str(observation))
            #action = RL.choose_action(str(observation))
            observation_,reward,done = env.step(action)
            total_reward_value = total_reward_value+reward
            RL.learn(str(observation),action,reward,str(observation_))
            observation = observation_
            if done:
                break
        print('{} episode over'.format(episode+1))
        print('average reward {}'.format(total_reward_value/(episode+1)))
        average_reward.append(total_reward_value/(episode+1))
    #print(RL.q_table)
    #RL.q_table.to_csv('3-2.csv',header=True,index=True)
    print(average_reward)
    env.destroy()
    #RL.q_table.to_clipboard()
    
if __name__=='__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env.after(2000,update)
    env.mainloop()
