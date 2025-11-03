import numpy as np
import matplotlib.pyplot as plt
import MDP
import RL2

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0


# Multi-arm bandit problems (3 arms with probabilities 0.3, 0.5 and 0.7)
T = np.array([[[1]],[[1]],[[1]]])
R = np.array([[0.3],[0.5],[0.7]])
discount = 0.999
mdp = MDP.MDP(T,R,discount)

# 實驗參數
nTrials = 1000  # 1000 次試驗
nIterations = 200  # 每次試驗 200 次迭代

# 儲存每次迭代的獎勵
rewards_epsilon = np.zeros([nTrials, nIterations])
rewards_thompson = np.zeros([nTrials, nIterations])
rewards_ucb = np.zeros([nTrials, nIterations])

print("開始進行 Bandit 實驗...")
print(f"總共 {nTrials} 次試驗，每次 {nIterations} 次迭代")
print("真實機率: [0.3, 0.5, 0.7]")
print("最優臂: 臂 2 (機率 0.7)\n")

# ==================== Epsilon-Greedy Bandit ====================
print("執行 Epsilon-Greedy Bandit...")
for trial in range(nTrials):
    if (trial + 1) % 100 == 0:
        print(f"  進度: {trial + 1}/{nTrials}")
    
    banditProblem = RL2.RL2(mdp, sampleBernoulli)
    
    # 重置統計資料
    counts = np.zeros(mdp.nActions)
    rewards_sum = np.zeros(mdp.nActions)
    
    for t in range(nIterations):
        epsilon = 1.0 / (t + 1)
        
        # Epsilon-greedy 選擇
        if np.random.rand() < epsilon:
            action = np.random.randint(mdp.nActions)
        else:
            empiricalMeans = rewards_sum / np.maximum(counts, 1)
            action = np.argmax(empiricalMeans)
        
        # 執行動作並獲得獎勵
        reward = sampleBernoulli(mdp.R[action, 0])
        counts[action] += 1
        rewards_sum[action] += reward
        
        # 記錄這次迭代的獎勵
        rewards_epsilon[trial, t] = reward

print("Epsilon-Greedy 完成!\n")

# ==================== Thompson Sampling Bandit ====================
print("執行 Thompson Sampling Bandit...")
for trial in range(nTrials):
    if (trial + 1) % 100 == 0:
        print(f"  進度: {trial + 1}/{nTrials}")
    
    banditProblem = RL2.RL2(mdp, sampleBernoulli)
    
    # 初始化 Beta 分布參數
    alpha = np.ones(mdp.nActions)
    beta = np.ones(mdp.nActions)
    
    for t in range(nIterations):
        # 從每個臂的 Beta 分布中抽樣
        sampled_means = np.random.beta(alpha, beta)
        action = np.argmax(sampled_means)
        
        # 執行動作並獲得獎勵
        reward = sampleBernoulli(mdp.R[action, 0])
        
        # 更新 Beta 分布參數
        if reward == 1:
            alpha[action] += 1
        else:
            beta[action] += 1
        
        # 記錄這次迭代的獎勵
        rewards_thompson[trial, t] = reward

print("Thompson Sampling 完成!\n")

# ==================== UCB Bandit ====================
print("執行 UCB Bandit...")
for trial in range(nTrials):
    if (trial + 1) % 100 == 0:
        print(f"  進度: {trial + 1}/{nTrials}")
    
    banditProblem = RL2.RL2(mdp, sampleBernoulli)
    
    # 重置統計資料
    counts = np.zeros(mdp.nActions)
    rewards_sum = np.zeros(mdp.nActions)
    
    # 首先每個臂至少拉一次
    for a in range(mdp.nActions):
        reward = sampleBernoulli(mdp.R[a, 0])
        counts[a] += 1
        rewards_sum[a] += reward
        rewards_ucb[trial, a] = reward
    
    # UCB 演算法
    for t in range(mdp.nActions, nIterations):
        empiricalMeans = rewards_sum / counts
        
        # 計算 UCB 值
        ucb_values = empiricalMeans + np.sqrt(2 * np.log(t + 1) / counts)
        
        # 選擇 UCB 值最高的臂
        action = np.argmax(ucb_values)
        
        # 執行動作並獲得獎勵
        reward = sampleBernoulli(mdp.R[action, 0])
        counts[action] += 1
        rewards_sum[action] += reward
        
        # 記錄這次迭代的獎勵
        rewards_ucb[trial, t] = reward

print("UCB 完成!\n")

# ==================== 計算平均並繪製圖表 ====================
print("計算平均獎勵並生成圖表...")

# 計算每次迭代的平均獎勵（基於 1000 次試驗）
avg_rewards_epsilon = np.mean(rewards_epsilon, axis=0)
avg_rewards_thompson = np.mean(rewards_thompson, axis=0)
avg_rewards_ucb = np.mean(rewards_ucb, axis=0)

# 繪製圖表
plt.figure(figsize=(12, 7))
plt.plot(range(nIterations), avg_rewards_epsilon, label='Epsilon-Greedy (ε=1/t)', linewidth=2, alpha=0.8)
plt.plot(range(nIterations), avg_rewards_thompson, label='Thompson Sampling (k=1)', linewidth=2, alpha=0.8)
plt.plot(range(nIterations), avg_rewards_ucb, label='UCB', linewidth=2, alpha=0.8)
plt.axhline(y=0.7, color='red', linestyle='--', label='Optimal (0.7)', linewidth=2, alpha=0.6)

plt.xlabel('Iteration #', fontsize=13)
plt.ylabel('Average Reward', fontsize=13)
plt.title('Bandit Algorithm Comparison\n(Average over 1000 trials)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.xlim(0, nIterations)
plt.ylim(0.25, 0.75)
plt.tight_layout()
plt.savefig('bandit_results.png', dpi=300, bbox_inches='tight')
print("圖表已儲存為 'bandit_results.png'")
plt.show()

# ==================== 統計分析 ====================
print("\n" + "="*60)
print("最終結果統計")
print("="*60)

# 計算最後 50 次迭代的平均獎勵
last_50_epsilon = np.mean(avg_rewards_epsilon[-50:])
last_50_thompson = np.mean(avg_rewards_thompson[-50:])
last_50_ucb = np.mean(avg_rewards_ucb[-50:])

print(f"\n最後 50 次迭代的平均獎勵:")
print(f"  Epsilon-Greedy: {last_50_epsilon:.4f}")
print(f"  Thompson Sampling: {last_50_thompson:.4f}")
print(f"  UCB: {last_50_ucb:.4f}")
print(f"  最優值: 0.7000")

# 計算累積遺憾 (Cumulative Regret)
optimal_reward = 0.7
cumulative_regret_epsilon = np.sum(optimal_reward - avg_rewards_epsilon)
cumulative_regret_thompson = np.sum(optimal_reward - avg_rewards_thompson)
cumulative_regret_ucb = np.sum(optimal_reward - avg_rewards_ucb)

print(f"\n累積遺憾 (Cumulative Regret):")
print(f"  Epsilon-Greedy: {cumulative_regret_epsilon:.2f}")
print(f"  Thompson Sampling: {cumulative_regret_thompson:.2f}")
print(f"  UCB: {cumulative_regret_ucb:.2f}")

# 計算收斂速度（達到 0.65 所需的迭代次數）
threshold = 0.65

def find_convergence(rewards, threshold):
    for i in range(len(rewards)):
        if np.mean(rewards[i:min(i+10, len(rewards))]) >= threshold:
            return i
    return len(rewards)

conv_epsilon = find_convergence(avg_rewards_epsilon, threshold)
conv_thompson = find_convergence(avg_rewards_thompson, threshold)
conv_ucb = find_convergence(avg_rewards_ucb, threshold)

print(f"\n收斂速度 (達到平均獎勵 {threshold} 所需迭代次數):")
print(f"  Epsilon-Greedy: {conv_epsilon} 次迭代")
print(f"  Thompson Sampling: {conv_thompson} 次迭代")
print(f"  UCB: {conv_ucb} 次迭代")

print("\n" + "="*60)
print("實驗完成!")
print("="*60)