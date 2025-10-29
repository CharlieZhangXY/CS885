import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        Q = initialQ.copy()
        
        # 學習率參數（可以設為遞減）
        alpha0 = 0.1  # 初始學習率
        
        for episode in range(nEpisodes):
            state = s0
            
            for step in range(nSteps):
                # 選擇動作：epsilon-greedy 或 Boltzmann exploration
                if np.random.rand() < epsilon:
                    # Epsilon exploration: 隨機選擇動作
                    action = np.random.randint(self.mdp.nActions)
                elif temperature > 0:
                    # Boltzmann exploration
                    # 計算每個動作的概率
                    q_values = Q[:, state]
                    exp_q = np.exp(q_values / temperature)
                    probs = exp_q / np.sum(exp_q)
                    action = np.random.choice(self.mdp.nActions, p=probs)
                else:
                    # Greedy: 選擇最佳動作
                    action = np.argmax(Q[:, state])
                
                # 執行動作，獲得獎勵和下一個狀態
                [reward, nextState] = self.sampleRewardAndNextState(state, action)
                
                # 學習率（可以隨時間遞減）
                # 使用訪問次數來調整學習率
                alpha = alpha0 / (1 + episode / 100.0)
                
                # Q-learning 更新規則
                # Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
                maxQnext = np.max(Q[:, nextState])
                Q[action, state] = Q[action, state] + alpha * (
                    reward + self.mdp.discount * maxQnext - Q[action, state]
                )
                
                # 移動到下一個狀態
                state = nextState
        
        # 從最終的 Q 函數提取策略
        policy = np.argmax(Q, axis=0)
        
        return [Q, policy]