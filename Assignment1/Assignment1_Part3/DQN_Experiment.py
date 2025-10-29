import gym
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
import collections
import random
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ========== 工具函數 ==========
class TorchHelper:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def f(self, x):
        return torch.tensor(x).float().to(self.device)
    
    def i(self, x):
        return torch.tensor(x).int().to(self.device)
    
    def l(self, x):
        return torch.tensor(x).long().to(self.device)

class ReplayBuffer:
    def __init__(self, N):
        self.buf = collections.deque(maxlen=N)
    
    def add(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))
    
    def sample(self, n, t):
        minibatch = random.sample(self.buf, n)
        S, A, R, S2, D = [], [], [], [], []
        for mb in minibatch:
            s, a, r, s2, d = mb
            S += [s]; A += [a]; R += [r]; S2 += [s2]; D += [d]
        return t.f(S), t.l(A), t.f(R), t.f(S2), t.i(D)

def seed_everything(seednum):
    random.seed(seednum)
    np.random.seed(seednum)
    torch.manual_seed(seednum)
    torch.cuda.manual_seed(seednum)

def play_episode_rb(env, policy, buf):
    states, actions, rewards = [], [], []
    states.append(env.reset())
    done = False
    while not done:
        action = policy(env, states[-1])
        actions.append(action)
        obs, reward, done, info = env.step(action)
        buf.add(states[-1], action, reward, obs, done)
        states.append(obs)
        rewards.append(reward)
    return states, actions, rewards

def play_episode(env, policy):
    states, actions, rewards = [], [], []
    states.append(env.reset())
    done = False
    while not done:
        action = policy(env, states[-1])
        actions.append(action)
        obs, reward, done, info = env.step(action)
        states.append(obs)
        rewards.append(reward)
    return states, actions, rewards

# ========== DQN 參數 ==========
SEEDS = [1, 2, 3, 4, 5]
t = TorchHelper()
DEVICE = t.device
OBS_N = 4
ACT_N = 2
GAMMA = 0.99
LEARNING_RATE = 5e-4
TRAIN_AFTER_EPISODES = 10
TRAIN_EPOCHS = 5
BUFSIZE = 10000
EPISODES = 300
TEST_EPISODES = 1
HIDDEN = 512
STARTING_EPSILON = 1.0
STEPS_MAX = 10000
EPSILON_END = 0.01

# ========== DQN 訓練函數 ==========
def create_everything(seed, minibatch_size):
    seed_everything(seed)
    env = gym.make("CartPole-v0")
    env.seed(seed)
    test_env = gym.make("CartPole-v0")
    test_env.seed(10+seed)
    buf = ReplayBuffer(BUFSIZE)
    Q = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N)
    ).to(DEVICE)
    Qt = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N)
    ).to(DEVICE)
    OPT = torch.optim.Adam(Q.parameters(), lr=LEARNING_RATE)
    return env, test_env, buf, Q, Qt, OPT

def update_target(target, source):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(p.data)

def update_networks(epi, buf, Q, Qt, OPT, minibatch_size, target_update_freq):
    S, A, R, S2, D = buf.sample(minibatch_size, t)
    qvalues = Q(S).gather(1, A.view(-1, 1)).squeeze()
    q2values = torch.max(Qt(S2), dim=1).values
    targets = R + GAMMA * q2values * (1-D)
    loss = torch.nn.MSELoss()(targets.detach(), qvalues)
    OPT.zero_grad()
    loss.backward()
    OPT.step()
    
    if epi % target_update_freq == 0:
        update_target(Qt, Q)
    
    return loss.item()

def train(seed, minibatch_size=10, target_update_freq=10):
    env, test_env, buf, Q, Qt, OPT = create_everything(seed, minibatch_size)
    EPSILON = STARTING_EPSILON
    testRs = []
    last25testRs = []
    
    def policy(env, obs):
        nonlocal EPSILON
        obs = t.f(obs).view(-1, OBS_N)
        if np.random.rand() < EPSILON:
            action = np.random.randint(ACT_N)
        else:
            qvalues = Q(obs)
            action = torch.argmax(qvalues).item()
        EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
        return action
    
    for epi in range(EPISODES):
        S, A, R = play_episode_rb(env, policy, buf)
        
        if epi >= TRAIN_AFTER_EPISODES:
            for tri in range(TRAIN_EPOCHS):
                update_networks(epi, buf, Q, Qt, OPT, minibatch_size, target_update_freq)
        
        Rews = []
        for epj in range(TEST_EPISODES):
            S, A, R = play_episode(test_env, policy)
            Rews += [sum(R)]
        testRs += [sum(Rews)/TEST_EPISODES]
        last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
    
    env.close()
    test_env.close()
    return last25testRs

def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label, linewidth=2)
    plt.fill_between(range(len(mean)), 
                     np.maximum(mean-std, 0), 
                     np.minimum(mean+std, 200), 
                     color=color, alpha=0.3)

# ========== 實驗 1: 目標網路更新頻率 ==========
def experiment1():
    print("=" * 50)
    print("實驗 1: 目標網路更新頻率的影響")
    print("=" * 50)
    
    target_freqs = [1, 10, 50, 100]
    colors = ['b', 'r', 'g', 'purple']
    
    plt.figure(figsize=(12, 6))
    
    for freq, color in zip(target_freqs, colors):
        print(f"\n測試目標網路更新頻率: {freq}")
        curves = []
        for seed in SEEDS:
            print(f"  種子 {seed}...")
            curve = train(seed, minibatch_size=10, target_update_freq=freq)
            curves.append(curve)
        plot_arrays(curves, color, f'更新頻率={freq}')
    
    plt.xlabel('訓練回合數', fontsize=12)
    plt.ylabel('最近25回合的平均累積獎勵', fontsize=12)
    plt.title('實驗1: 目標網路更新頻率對DQN性能的影響', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiment1_target_network.png', dpi=300)
    plt.show()

# ========== 實驗 2: Mini-batch 大小 ==========
def experiment2():
    print("\n" + "=" * 50)
    print("實驗 2: Mini-batch 大小的影響")
    print("=" * 50)
    
    batch_sizes = [1, 10, 50, 100]
    colors = ['b', 'r', 'g', 'purple']
    
    plt.figure(figsize=(12, 6))
    
    for batch_size, color in zip(batch_sizes, colors):
        print(f"\nTesting mini-batch 大小: {batch_size}")
        curves = []
        for seed in SEEDS:
            print(f"  種子 {seed}...")
            curve = train(seed, minibatch_size=batch_size, target_update_freq=10)
            curves.append(curve)
        plot_arrays(curves, color, f'Batch大小={batch_size}')
    
    plt.xlabel('訓練回合數', fontsize=12)
    plt.ylabel('最近25回合的平均累積獎勵', fontsize=12)
    plt.title('實驗2: Mini-batch大小對DQN性能的影響', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiment2_minibatch.png', dpi=300)
    plt.show()

# ========== 主程式 ==========
if __name__ == "__main__":
    # 執行實驗 1
    experiment1()
    
    # 執行實驗 2
    experiment2()
    
    print("\n" + "=" * 50)
    print("所有實驗完成！")
    print("=" * 50)