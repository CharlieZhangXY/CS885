import numpy as np
import matplotlib.pyplot as plt
import MDP
import RL2

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
matplotlib.rcParams['axes.unicode_minus'] = False

''' Construct a simple maze MDP

  Grid world layout:

  ---------------------
  |  0 |  1 |  2 |  3 |
  ---------------------
  |  4 |  5 |  6 |  7 |
  ---------------------
  |  8 |  9 | 10 | 11 |
  ---------------------
  | 12 | 13 | 14 | 15 |
  ---------------------

  Goal state: 14 
  Bad state: 9
  End state: 16

  The end state is an absorbing state that the agent transitions 
  to after visiting the goal state.

  There are 17 states in total (including the end state) 
  and 4 actions (up, down, left, right).'''

# Transition function: |A| x |S| x |S'| array
T = np.zeros([4,17,17])
a = 0.8;  # intended move
b = 0.1;  # lateral move

# up (a = 0)
T[0,0,0] = a+b;
T[0,0,1] = b;

T[0,1,0] = b;
T[0,1,1] = a;
T[0,1,2] = b;

T[0,2,1] = b;
T[0,2,2] = a;
T[0,2,3] = b;

T[0,3,2] = b;
T[0,3,3] = a+b;

T[0,4,4] = b;
T[0,4,0] = a;
T[0,4,5] = b;

T[0,5,4] = b;
T[0,5,1] = a;
T[0,5,6] = b;

T[0,6,5] = b;
T[0,6,2] = a;
T[0,6,7] = b;

T[0,7,6] = b;
T[0,7,3] = a;
T[0,7,7] = b;

T[0,8,8] = b;
T[0,8,4] = a;
T[0,8,9] = b;

T[0,9,8] = b;
T[0,9,5] = a;
T[0,9,10] = b;

T[0,10,9] = b;
T[0,10,6] = a;
T[0,10,11] = b;

T[0,11,10] = b;
T[0,11,7] = a;
T[0,11,11] = b;

T[0,12,12] = b;
T[0,12,8] = a;
T[0,12,13] = b;

T[0,13,12] = b;
T[0,13,9] = a;
T[0,13,14] = b;

T[0,14,16] = 1;

T[0,15,11] = a;
T[0,15,14] = b;
T[0,15,15] = b;

T[0,16,16] = 1;

# down (a = 1)
T[1,0,0] = b;
T[1,0,4] = a;
T[1,0,1] = b;

T[1,1,0] = b;
T[1,1,5] = a;
T[1,1,2] = b;

T[1,2,1] = b;
T[1,2,6] = a;
T[1,2,3] = b;

T[1,3,2] = b;
T[1,3,7] = a;
T[1,3,3] = b;

T[1,4,4] = b;
T[1,4,8] = a;
T[1,4,5] = b;

T[1,5,4] = b;
T[1,5,9] = a;
T[1,5,6] = b;

T[1,6,5] = b;
T[1,6,10] = a;
T[1,6,7] = b;

T[1,7,6] = b;
T[1,7,11] = a;
T[1,7,7] = b;

T[1,8,8] = b;
T[1,8,12] = a;
T[1,8,9] = b;

T[1,9,8] = b;
T[1,9,13] = a;
T[1,9,10] = b;

T[1,10,9] = b;
T[1,10,14] = a;
T[1,10,11] = b;

T[1,11,10] = b;
T[1,11,15] = a;
T[1,11,11] = b;

T[1,12,12] = a+b;
T[1,12,13] = b;

T[1,13,12] = b;
T[1,13,13] = a;
T[1,13,14] = b;

T[1,14,16] = 1;

T[1,15,14] = b;
T[1,15,15] = a+b;

T[1,16,16] = 1;

# left (a = 2)
T[2,0,0] = a+b;
T[2,0,4] = b;

T[2,1,1] = b;
T[2,1,0] = a;
T[2,1,5] = b;

T[2,2,2] = b;
T[2,2,1] = a;
T[2,2,6] = b;

T[2,3,3] = b;
T[2,3,2] = a;
T[2,3,7] = b;

T[2,4,0] = b;
T[2,4,4] = a;
T[2,4,8] = b;

T[2,5,1] = b;
T[2,5,4] = a;
T[2,5,9] = b;

T[2,6,2] = b;
T[2,6,5] = a;
T[2,6,10] = b;

T[2,7,3] = b;
T[2,7,6] = a;
T[2,7,11] = b;

T[2,8,4] = b;
T[2,8,8] = a;
T[2,8,12] = b;

T[2,9,5] = b;
T[2,9,8] = a;
T[2,9,13] = b;

T[2,10,6] = b;
T[2,10,9] = a;
T[2,10,14] = b;

T[2,11,7] = b;
T[2,11,10] = a;
T[2,11,15] = b;

T[2,12,8] = b;
T[2,12,12] = a+b;

T[2,13,9] = b;
T[2,13,12] = a;
T[2,13,13] = b;

T[2,14,16] = 1;

T[2,15,11] = a;
T[2,15,14] = b;
T[2,15,15] = b;

T[2,16,16] = 1;

# right (a = 3)
T[3,0,0] = b;
T[3,0,1] = a;
T[3,0,4] = b;

T[3,1,1] = b;
T[3,1,2] = a;
T[3,1,5] = b;

T[3,2,2] = b;
T[3,2,3] = a;
T[3,2,6] = b;

T[3,3,3] = a+b;
T[3,3,7] = b;

T[3,4,0] = b;
T[3,4,5] = a;
T[3,4,8] = b;

T[3,5,1] = b;
T[3,5,6] = a;
T[3,5,9] = b;

T[3,6,2] = b;
T[3,6,7] = a;
T[3,6,10] = b;

T[3,7,3] = b;
T[3,7,7] = a;
T[3,7,11] = b;

T[3,8,4] = b;
T[3,8,9] = a;
T[3,8,12] = b;

T[3,9,5] = b;
T[3,9,10] = a;
T[3,9,13] = b;

T[3,10,6] = b;
T[3,10,11] = a;
T[3,10,14] = b;

T[3,11,7] = b;
T[3,11,11] = a;
T[3,11,15] = b;

T[3,12,8] = b;
T[3,12,13] = a;
T[3,12,12] = b;

T[3,13,9] = b;
T[3,13,14] = a;
T[3,13,13] = b;

T[3,14,16] = 1;

T[3,15,11] = b;
T[3,15,15] = a+b;

T[3,16,16] = 1;

# Reward function: |A| x |S| array
R = -1 * np.ones([4,17]);

# set rewards
R[:,14] = 100;  # goal state
R[:,9] = -70;   # bad state
R[:,16] = 0;    # end state

# Discount factor: scalar in [0,1)
discount = 0.95
        
# MDP object
mdp = MDP.MDP(T,R,discount)

# 實驗參數
nTrials = 100  # 100 次試驗
nEpisodes = 200  # 200 個回合
nSteps = 100  # 每回合 100 步
epsilon = 0.05

# 儲存每次試驗的累積折扣獎勵
cumulative_rewards_mb = np.zeros([nTrials, nEpisodes])
cumulative_rewards_ql = np.zeros([nTrials, nEpisodes])

print("開始進行迷宮實驗...")
print(f"總共 {nTrials} 次試驗，每次 {nEpisodes} 個回合，每回合 {nSteps} 步")
print(f"Epsilon: {epsilon}")
print(f"折扣因子: {discount}\n")

# ==================== Model-Based RL ====================
print("執行 Model-Based RL...")
for trial in range(nTrials):
    if (trial + 1) % 10 == 0:
        print(f"  進度: {trial + 1}/{nTrials}")
    
    rlProblem = RL2.RL2(mdp, np.random.normal)
    
    # 初始化
    T_est = np.ones([mdp.nActions, mdp.nStates, mdp.nStates]) / mdp.nStates
    R_est = np.zeros([mdp.nActions, mdp.nStates])
    
    visit_count = np.zeros([mdp.nActions, mdp.nStates])
    transition_count = np.zeros([mdp.nActions, mdp.nStates, mdp.nStates])
    reward_sum = np.zeros([mdp.nActions, mdp.nStates])
    
    policy = np.zeros(mdp.nStates, int)
    
    for episode in range(nEpisodes):
        state = 0  # 從狀態 0 開始
        cumulative_reward = 0
        discount_factor = 1.0
        
        for step in range(nSteps):
            # Epsilon-greedy 行動選擇
            if np.random.rand() < epsilon:
                action = np.random.randint(mdp.nActions)
            else:
                action = policy[state]
            
            # 執行行動並觀察結果
            reward = np.random.normal(mdp.R[action, state])
            cumProb = np.cumsum(mdp.T[action, state, :])
            nextState = np.where(cumProb >= np.random.rand(1))[0][0]
            
            # 累積折扣獎勵
            cumulative_reward += discount_factor * reward
            discount_factor *= discount
            
            # 更新統計資料
            visit_count[action, state] += 1
            transition_count[action, state, nextState] += 1
            reward_sum[action, state] += reward
            
            # 更新估計的轉移函數
            T_est[action, state, :] = transition_count[action, state, :] / visit_count[action, state]
            
            # 更新估計的獎勵函數
            R_est[action, state] = reward_sum[action, state] / visit_count[action, state]
            
            state = nextState
        
        # 記錄這個回合的累積折扣獎勵
        cumulative_rewards_mb[trial, episode] = cumulative_reward
        
        # 使用估計的 MDP 更新策略
        estimated_mdp = MDP.MDP(T_est, R_est, mdp.discount)
        V = estimated_mdp.valueIteration(initialV=np.zeros(mdp.nStates), nIterations=100)[0]
        policy = estimated_mdp.extractPolicy(V)

print("Model-Based RL 完成!\n")

# ==================== Q-Learning ====================
print("執行 Q-Learning...")
for trial in range(nTrials):
    if (trial + 1) % 10 == 0:
        print(f"  進度: {trial + 1}/{nTrials}")
    
    rlProblem = RL2.RL2(mdp, np.random.normal)
    
    # 初始化 Q 函數
    Q = np.zeros([mdp.nActions, mdp.nStates])
    
    # 學習率
    alpha = 0.1
    
    for episode in range(nEpisodes):
        state = 0  # 從狀態 0 開始
        cumulative_reward = 0
        discount_factor = 1.0
        
        for step in range(nSteps):
            # Epsilon-greedy 行動選擇
            if np.random.rand() < epsilon:
                action = np.random.randint(mdp.nActions)
            else:
                action = np.argmax(Q[:, state])
            
            # 執行行動並觀察結果
            reward = np.random.normal(mdp.R[action, state])
            cumProb = np.cumsum(mdp.T[action, state, :])
            nextState = np.where(cumProb >= np.random.rand(1))[0][0]
            
            # 累積折扣獎勵
            cumulative_reward += discount_factor * reward
            discount_factor *= discount
            
            # Q-Learning 更新
            Q[action, state] = Q[action, state] + alpha * (
                reward + mdp.discount * np.max(Q[:, nextState]) - Q[action, state]
            )
            
            state = nextState
        
        # 記錄這個回合的累積折扣獎勵
        cumulative_rewards_ql[trial, episode] = cumulative_reward

print("Q-Learning 完成!\n")

# ==================== 計算平均並繪製圖表 ====================
print("計算平均獎勵並生成圖表...")

# 計算每個回合的平均累積折扣獎勵（基於 100 次試驗）
avg_rewards_mb = np.mean(cumulative_rewards_mb, axis=0)
avg_rewards_ql = np.mean(cumulative_rewards_ql, axis=0)

# 繪製圖表
plt.figure(figsize=(12, 7))
plt.plot(range(nEpisodes), avg_rewards_mb, label='Model-Based RL', linewidth=2, alpha=0.8)
plt.plot(range(nEpisodes), avg_rewards_ql, label='Q-Learning', linewidth=2, alpha=0.8)

plt.xlabel('回合編號 (Episode #)', fontsize=13)
plt.ylabel('平均累積折扣獎勵', fontsize=13)
plt.title('迷宮問題：Model-Based RL vs Q-Learning\n(基於 100 次試驗的平均)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.xlim(0, nEpisodes)
plt.tight_layout()
plt.savefig('maze_comparison.png', dpi=300, bbox_inches='tight')
print("圖表已儲存為 'maze_comparison.png'")
plt.show()

# ==================== 統計分析 ====================
print("\n" + "="*70)
print("最終結果統計")
print("="*70)

# 計算前 50 個回合和後 50 個回合的平均
first_50_mb = np.mean(avg_rewards_mb[:50])
last_50_mb = np.mean(avg_rewards_mb[-50:])
first_50_ql = np.mean(avg_rewards_ql[:50])
last_50_ql = np.mean(avg_rewards_ql[-50:])

print(f"\n前 50 個回合的平均累積折扣獎勵:")
print(f"  Model-Based RL: {first_50_mb:.2f}")
print(f"  Q-Learning: {first_50_ql:.2f}")

print(f"\n後 50 個回合的平均累積折扣獎勵:")
print(f"  Model-Based RL: {last_50_mb:.2f}")
print(f"  Q-Learning: {last_50_ql:.2f}")

# 計算總累積獎勵
total_mb = np.sum(avg_rewards_mb)
total_ql = np.sum(avg_rewards_ql)

print(f"\n總累積獎勵 (200 個回合):")
print(f"  Model-Based RL: {total_mb:.2f}")
print(f"  Q-Learning: {total_ql:.2f}")

# 計算收斂速度
threshold = 0

def find_convergence(rewards, threshold):
    for i in range(len(rewards)):
        if np.mean(rewards[i:min(i+20, len(rewards))]) >= threshold:
            return i
    return len(rewards)

conv_mb = find_convergence(avg_rewards_mb, threshold)
conv_ql = find_convergence(avg_rewards_ql, threshold)

print(f"\n收斂速度 (達到平均獎勵 {threshold} 所需回合數):")
print(f"  Model-Based RL: {conv_mb} 個回合")
print(f"  Q-Learning: {conv_ql} 個回合")

print("\n" + "="*70)
print("實驗完成!")
print("="*70)