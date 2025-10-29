import numpy as np
import matplotlib.pyplot as plt
import MDP
import RL

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# 構建迷宮 MDP（與 Part I 相同）
T = np.zeros([4,17,17])
a = 0.8
b = 0.1

# up (a = 0)
T[0,0,0] = a+b; T[0,0,1] = b
T[0,1,0] = b; T[0,1,1] = a; T[0,1,2] = b
T[0,2,1] = b; T[0,2,2] = a; T[0,2,3] = b
T[0,3,2] = b; T[0,3,3] = a+b
T[0,4,4] = b; T[0,4,0] = a; T[0,4,5] = b
T[0,5,4] = b; T[0,5,1] = a; T[0,5,6] = b
T[0,6,5] = b; T[0,6,2] = a; T[0,6,7] = b
T[0,7,6] = b; T[0,7,3] = a; T[0,7,7] = b
T[0,8,8] = b; T[0,8,4] = a; T[0,8,9] = b
T[0,9,8] = b; T[0,9,5] = a; T[0,9,10] = b
T[0,10,9] = b; T[0,10,6] = a; T[0,10,11] = b
T[0,11,10] = b; T[0,11,7] = a; T[0,11,11] = b
T[0,12,12] = b; T[0,12,8] = a; T[0,12,13] = b
T[0,13,12] = b; T[0,13,9] = a; T[0,13,14] = b
T[0,14,16] = 1
T[0,15,11] = a; T[0,15,14] = b; T[0,15,15] = b
T[0,16,16] = 1

# down (a = 1)
T[1,0,0] = b; T[1,0,4] = a; T[1,0,1] = b
T[1,1,0] = b; T[1,1,5] = a; T[1,1,2] = b
T[1,2,1] = b; T[1,2,6] = a; T[1,2,3] = b
T[1,3,2] = b; T[1,3,7] = a; T[1,3,3] = b
T[1,4,4] = b; T[1,4,8] = a; T[1,4,5] = b
T[1,5,4] = b; T[1,5,9] = a; T[1,5,6] = b
T[1,6,5] = b; T[1,6,10] = a; T[1,6,7] = b
T[1,7,6] = b; T[1,7,11] = a; T[1,7,7] = b
T[1,8,8] = b; T[1,8,12] = a; T[1,8,9] = b
T[1,9,8] = b; T[1,9,13] = a; T[1,9,10] = b
T[1,10,9] = b; T[1,10,14] = a; T[1,10,11] = b
T[1,11,10] = b; T[1,11,15] = a; T[1,11,11] = b
T[1,12,12] = a+b; T[1,12,13] = b
T[1,13,12] = b; T[1,13,13] = a; T[1,13,14] = b
T[1,14,16] = 1
T[1,15,14] = b; T[1,15,15] = a+b
T[1,16,16] = 1

# left (a = 2)
T[2,0,0] = a+b; T[2,0,4] = b
T[2,1,1] = b; T[2,1,0] = a; T[2,1,5] = b
T[2,2,2] = b; T[2,2,1] = a; T[2,2,6] = b
T[2,3,3] = b; T[2,3,2] = a; T[2,3,7] = b
T[2,4,0] = b; T[2,4,4] = a; T[2,4,8] = b
T[2,5,1] = b; T[2,5,4] = a; T[2,5,9] = b
T[2,6,2] = b; T[2,6,5] = a; T[2,6,10] = b
T[2,7,3] = b; T[2,7,6] = a; T[2,7,11] = b
T[2,8,4] = b; T[2,8,8] = a; T[2,8,12] = b
T[2,9,5] = b; T[2,9,8] = a; T[2,9,13] = b
T[2,10,6] = b; T[2,10,9] = a; T[2,10,14] = b
T[2,11,7] = b; T[2,11,10] = a; T[2,11,15] = b
T[2,12,8] = b; T[2,12,12] = a+b
T[2,13,9] = b; T[2,13,12] = a; T[2,13,13] = b
T[2,14,16] = 1
T[2,15,11] = a; T[2,15,14] = b; T[2,15,15] = b
T[2,16,16] = 1

# right (a = 3)
T[3,0,0] = b; T[3,0,1] = a; T[3,0,4] = b
T[3,1,1] = b; T[3,1,2] = a; T[3,1,5] = b
T[3,2,2] = b; T[3,2,3] = a; T[3,2,6] = b
T[3,3,3] = a+b; T[3,3,7] = b
T[3,4,0] = b; T[3,4,5] = a; T[3,4,8] = b
T[3,5,1] = b; T[3,5,6] = a; T[3,5,9] = b
T[3,6,2] = b; T[3,6,7] = a; T[3,6,10] = b
T[3,7,3] = b; T[3,7,7] = a; T[3,7,11] = b
T[3,8,4] = b; T[3,8,9] = a; T[3,8,12] = b
T[3,9,5] = b; T[3,9,10] = a; T[3,9,13] = b
T[3,10,6] = b; T[3,10,11] = a; T[3,10,14] = b
T[3,11,7] = b; T[3,11,11] = a; T[3,11,15] = b
T[3,12,8] = b; T[3,12,13] = a; T[3,12,12] = b
T[3,13,9] = b; T[3,13,14] = a; T[3,13,13] = b
T[3,14,16] = 1
T[3,15,11] = b; T[3,15,15] = a+b
T[3,16,16] = 1

# 獎勵函數
R = -1 * np.ones([4,17])
R[:,14] = 100  # goal state
R[:,9] = -70   # bad state
R[:,16] = 0    # end state

discount = 0.95
mdp = MDP.MDP(T,R,discount)

def calculate_episode_reward(mdp, Q, s0, nSteps, epsilon):
    """計算一個 episode 的累積折扣獎勵"""
    state = s0
    cumulative_reward = 0
    discount_factor = 1.0
    
    for step in range(nSteps):
        # 使用 epsilon-greedy 策略選擇動作
        if np.random.rand() < epsilon:
            action = np.random.randint(mdp.nActions)
        else:
            action = np.argmax(Q[:, state])
        
        # 獲取獎勵和下一個狀態
        reward = mdp.R[action, state]
        cumProb = np.cumsum(mdp.T[action, state, :])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        
        # 累積折扣獎勵
        cumulative_reward += discount_factor * reward
        discount_factor *= mdp.discount
        
        state = nextState
    
    return cumulative_reward

def run_qlearning_experiment(mdp, epsilon_value, nTrials=100, nEpisodes=200, nSteps=100):
    """運行 Q-learning 實驗，記錄每個 episode 的累積獎勵"""
    # 儲存所有 trial 的結果
    all_rewards = np.zeros((nTrials, nEpisodes))
    
    for trial in range(nTrials):
        if trial % 10 == 0:
            print(f"  Trial {trial}/{nTrials}...")
        
        rlProblem = RL.RL(mdp, np.random.normal)
        Q = np.zeros([mdp.nActions, mdp.nStates])
        
        # 對每個 episode 進行訓練並記錄獎勵
        for episode in range(nEpisodes):
            # 訓練一個 episode
            alpha = 0.1 / (1 + episode / 100.0)
            state = 0  # s0
            
            for step in range(nSteps):
                # 選擇動作
                if np.random.rand() < epsilon_value:
                    action = np.random.randint(mdp.nActions)
                else:
                    action = np.argmax(Q[:, state])
                
                # 執行動作
                reward = np.random.normal(mdp.R[action, state])
                cumProb = np.cumsum(mdp.T[action, state, :])
                nextState = np.where(cumProb >= np.random.rand(1))[0][0]
                
                # Q-learning 更新
                maxQnext = np.max(Q[:, nextState])
                Q[action, state] = Q[action, state] + alpha * (
                    reward + mdp.discount * maxQnext - Q[action, state]
                )
                
                state = nextState
            
            # 計算這個 episode 的累積獎勵（評估當前策略）
            all_rewards[trial, episode] = calculate_episode_reward(mdp, Q, 0, nSteps, epsilon_value)
    
    # 返回平均獎勵
    return np.mean(all_rewards, axis=0), Q

# 運行實驗
print("開始 Q-learning 實驗...")
epsilon_values = [0.05, 0.1, 0.3, 0.5]
results = {}
final_Qs = {}
final_policies = {}

for eps in epsilon_values:
    print(f"\n執行 epsilon = {eps} 的實驗...")
    avg_rewards, final_Q = run_qlearning_experiment(mdp, eps, nTrials=100, nEpisodes=200, nSteps=100)
    results[eps] = avg_rewards
    final_Qs[eps] = final_Q
    final_policies[eps] = np.argmax(final_Q, axis=0)

# 繪製結果
plt.figure(figsize=(12, 7))
action_names = ['上(Up)', '下(Down)', '左(Left)', '右(Right)']

for eps in epsilon_values:
    plt.plot(range(200), results[eps], label=f'ε = {eps}', linewidth=2)

plt.xlabel('Episode 編號', fontsize=12)
plt.ylabel('平均累積折扣獎勵（100次試驗）', fontsize=12)
plt.title('Q-learning: 不同探索率下的學習曲線', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('qlearning_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 輸出最終策略和 Q 值分析
print("\n" + "="*70)
print("最終結果分析")
print("="*70)

for eps in epsilon_values:
    print(f"\nε = {eps}:")
    print(f"  最終平均獎勵: {results[eps][-1]:.2f}")
    
    print(f"\n  最終策略:")
    for s in range(min(17, mdp.nStates)):
        print(f"    State {s:2d}: Action {final_policies[eps][s]} ({action_names[final_policies[eps][s]]})")
    
    print(f"\n  部分 Q 值 (前 5 個狀態):")
    for s in range(min(5, mdp.nStates)):
        print(f"    State {s}: {final_Qs[eps][:, s]}")

print("\n" + "="*70)
print("實驗完成！圖表已儲存為 'qlearning_results.png'")
print("="*70)