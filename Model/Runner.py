import numpy as np
import os

from Model.learner import Learner
from Env import EpEnv



class Runner:
    def __init__(self, args, device):
        self.args = args
        self.agents = Learner(args)


        self.device = device
        self.env = EpEnv( args= self.args, agents= self.agents)


    def run(self,total_episodes):

        if not self.args.evaluate:
            for _ in range(total_episodes): #开始训练
                self.env.generate_ep_episode()

        mean_rate, episode_rewards, Mean_T_violation, Mean_T_violation_offset = self.evaluate()
        print(f'cost_rate is {mean_rate * 100} %')
        print('episode_rewards is ', episode_rewards)
        print(f'Mean_T_violation is {Mean_T_violation * 100} %')
        print(f'Mean_T_violation_offset is  {Mean_T_violation_offset}')
        self.agents.close()

    def evaluate(self):
        mean_rate = []
        episode_rewards = []
        Mean_T_violation = []
        Mean_T_violation_offset = []
        for epoch in range(self.args.evaluate_epoch):
            print(20*'-',f"Start evaluate {epoch}",20*'-')
            episode_reward, cost_rate,T_violation ,T_violation_offset = self.env.generate_ep_episode(evaluate = True)
            episode_rewards.append(episode_reward)
            mean_rate.append(cost_rate)
            Mean_T_violation.append(T_violation)
            Mean_T_violation_offset.append(T_violation_offset)
        mean_rate = np.mean(mean_rate)
        episode_rewards = np.mean(episode_rewards)
        Mean_T_violation = np.mean(Mean_T_violation)
        Mean_T_violation_offset = np.mean(Mean_T_violation_offset)
        return mean_rate, episode_rewards, Mean_T_violation, Mean_T_violation_offset


