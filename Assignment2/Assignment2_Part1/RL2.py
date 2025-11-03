import numpy as np
import MDP

class RL2:
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

    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):
        '''Model-based Reinforcement Learning with epsilon greedy 
        exploration.  This function should use value iteration,
        policy iteration or modified policy iteration to update the policy at each step

        Inputs:
        s0 -- initial state
        defaultT -- default transition function when a state-action pair has not been vsited
        initialR -- initial estimate of the reward function
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random

        Outputs: 
        V -- final value function
        policy -- final policy
        '''

        # 初始化估計的轉移函數和獎勵函數
        T_est = defaultT.copy()
        R_est = initialR.copy()
        
        # 記錄訪問次數
        visit_count = np.zeros([self.mdp.nActions, self.mdp.nStates])
        transition_count = np.zeros([self.mdp.nActions, self.mdp.nStates, self.mdp.nStates])
        reward_sum = np.zeros([self.mdp.nActions, self.mdp.nStates])
        
        # 初始化策略
        policy = np.zeros(self.mdp.nStates, int)
        
        for episode in range(nEpisodes):
            state = s0
            
            for step in range(nSteps):
                # Epsilon-greedy 行動選擇
                if np.random.rand() < epsilon:
                    action = np.random.randint(self.mdp.nActions)
                else:
                    action = policy[state]
                
                # 執行行動並觀察結果
                [reward, nextState] = self.sampleRewardAndNextState(state, action)
                
                # 更新統計資料
                visit_count[action, state] += 1
                transition_count[action, state, nextState] += 1
                reward_sum[action, state] += reward
                
                # 更新估計的轉移函數
                T_est[action, state, :] = transition_count[action, state, :] / visit_count[action, state]
                
                # 更新估計的獎勵函數
                R_est[action, state] = reward_sum[action, state] / visit_count[action, state]
                
                state = nextState
            
            # 使用估計的 MDP 更新策略
            estimated_mdp = MDP.MDP(T_est, R_est, self.mdp.discount)
            V = estimated_mdp.valueIteration(initialV=np.zeros(self.mdp.nStates), nIterations=100)[0]
            policy = estimated_mdp.extractPolicy(V)
        
        # 計算最終價值函數
        estimated_mdp = MDP.MDP(T_est, R_est, self.mdp.discount)
        V = estimated_mdp.valueIteration(initialV=np.zeros(self.mdp.nStates), nIterations=100)[0]

        return [V, policy]

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor).  Use epsilon = 1 / # of iterations.

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # 初始化
        counts = np.zeros(self.mdp.nActions)
        rewards = np.zeros(self.mdp.nActions)
        
        for t in range(1, nIterations + 1):
            epsilon = 1.0 / t
            
            # Epsilon-greedy 選擇
            if np.random.rand() < epsilon:
                action = np.random.randint(self.mdp.nActions)
            else:
                # 選擇當前最佳臂
                empiricalMeans = rewards / np.maximum(counts, 1)
                action = np.argmax(empiricalMeans)
            
            # 拉動選擇的臂並觀察獎勵
            reward = self.sampleReward(self.mdp.R[action, 0])
            counts[action] += 1
            rewards[action] += reward
        
        empiricalMeans = rewards / np.maximum(counts, 1)
        return empiricalMeans

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # 初始化 Beta 分布參數
        alpha = prior[:, 0].copy()
        beta = prior[:, 1].copy()
        
        counts = np.zeros(self.mdp.nActions)
        rewards = np.zeros(self.mdp.nActions)
        
        for t in range(nIterations):
            # 從每個臂的 Beta 分布中抽樣 k 次
            sampled_means = np.zeros(self.mdp.nActions)
            for a in range(self.mdp.nActions):
                samples = np.random.beta(alpha[a], beta[a], k)
                sampled_means[a] = np.mean(samples)
            
            # 選擇抽樣平均值最高的臂
            action = np.argmax(sampled_means)
            
            # 拉動選擇的臂並觀察獎勵
            reward = self.sampleReward(self.mdp.R[action, 0])
            counts[action] += 1
            rewards[action] += reward
            
            # 更新 Beta 分布參數
            if reward == 1:
                alpha[action] += 1
            else:
                beta[action] += 1
        
        empiricalMeans = rewards / np.maximum(counts, 1)
        return empiricalMeans

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # 初始化
        counts = np.zeros(self.mdp.nActions)
        rewards = np.zeros(self.mdp.nActions)
        
        # 首先每個臂至少拉一次
        for a in range(self.mdp.nActions):
            reward = self.sampleReward(self.mdp.R[a, 0])
            counts[a] += 1
            rewards[a] += reward
        
        # UCB 演算法
        for t in range(self.mdp.nActions + 1, nIterations + 1):
            empiricalMeans = rewards / counts
            
            # 計算 UCB 值
            ucb_values = empiricalMeans + np.sqrt(2 * np.log(t) / counts)
            
            # 選擇 UCB 值最高的臂
            action = np.argmax(ucb_values)
            
            # 拉動選擇的臂並觀察獎勵
            reward = self.sampleReward(self.mdp.R[action, 0])
            counts[action] += 1
            rewards[action] += reward
        
        empiricalMeans = rewards / counts
        return empiricalMeans