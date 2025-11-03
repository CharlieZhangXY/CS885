from math import log
import gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch
import torch, random
from torch import nn
import copy
import tqdm
import matplotlib.pyplot as plt
import warnings
import argparse

warnings.filterwarnings("ignore")

# Proximal Policy Optimization
# Slide 14
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-fall21/slides/cs885-module1.pdf

parser = argparse.ArgumentParser()

#either:
# cartpole - default cartpole environment
# mountain_car - default mountain car environment
# mountain_car_mod - mountain car environment with modified reward
parser.add_argument('--mode', type=str, default="cartpole") 

args = parser.parse_args()


# Constants
SEED = 1
t = utils.torch.TorchHelper()
DEVICE = t.device

#for cartpole
if args.mode == "cartpole":
    OBS_N = 4               # State space size
    ACT_N = 2               # Action space size
    ENV_NAME = "CartPole-v0"
    GAMMA = 1.0               # Discount factor in episodic reward objective
    LEARNING_RATE1 = 5e-4     # Learning rate for value optimizer
    LEARNING_RATE2 = 5e-4     # Learning rate for actor optimizer
    POLICY_TRAIN_ITERS = 10   # Training epochs

elif "mountain_car" in args.mode:
    OBS_N = 2
    ACT_N = 3
    ENV_NAME = "MountainCar-v0"
    GAMMA = 0.9               # Discount factor in episodic reward objective
    LEARNING_RATE1 = 1e-3     # Learning rate for value optimizer
    LEARNING_RATE2 = 1e-3     # Learning rate for actor optimizer
    POLICY_TRAIN_ITERS = 5   # Training epochs
   
EPOCHS = 150              # Total number of epochs to learn over
EPISODES_PER_EPOCH = 1    # Episodes per epoch
TEST_EPISODES = 10        # Test episodes
HIDDEN = 32               # Hidden size
CLIP_PARAM = 0.1          # Clip parameter


# Create environment
utils.seed.seed(SEED)
env = gym.make(ENV_NAME)
env.seed(SEED)

# Networks
V = torch.nn.Sequential(
    torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, 1)
).to(DEVICE)
pi = torch.nn.Sequential(
    torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, ACT_N)
).to(DEVICE)

# Optimizers
OPT1 = torch.optim.Adam(V.parameters(), lr = LEARNING_RATE1)
OPT2 = torch.optim.Adam(pi.parameters(), lr = LEARNING_RATE2)

# Policy
def policy(env, obs):
    probs = torch.nn.Softmax(dim=-1)(pi(t.f(obs)))
    return np.random.choice(ACT_N, p = probs.cpu().detach().numpy())

# Training function
# S = tensor of states observed in the episode/ batch of episodes
# A = tensor of actions taken in episode/ batch of episodes
# return = tensor where nth element is \sum^{T-n}_0 gamma^n * reward (return at step n of episode)
# old_log_probs = tensor of pi(a | s) for policy the trajectory was collected under
def train(S,A,returns, old_log_probs):

    ###############################
    # YOUR CODE HERE:

    # --- 1. 訓練 Value Function (Critic) ---
    # 根據 REINFORCE_Baseline.py 的要求，我們使用 MSE Loss 來訓練 V(s)
    # V(s) 的目標是逼近實際的 return (G_t)
    
    # 取得目前 Value Network 對所有狀態 S 的預測值
    V_s = V(S).flatten()
    
    # 計算 V(s) 與實際 returns 之間的 Mean Squared Error Loss
    V_loss = nn.MSELoss()(V_s, returns)
    
    # 更新 Value Network
    OPT1.zero_grad()
    V_loss.backward()
    OPT1.step()

    # --- 2. 計算 Advantage ---
    # Advantage A(s, a) = G_t - V(s)
    # 我們使用更新後的 V(s) 來計算 Advantage
    # .detach() 是為了確保在更新 Policy 時，梯度不會反向傳播回 Value Network
    A_t = returns - V(S).flatten().detach()
    
    # (重要) 將 Advantage 標準化 (Standardization)
    # 這能顯著提高 PPO 的穩定性
    A_t = (A_t - A_t.mean()) / (A_t.std() + 1e-8)


    # --- 3. 訓練 Policy (Actor) - PPO-Clip ---
    # 執行 POLICY_TRAIN_ITERS 次梯度上升
    for i in range(POLICY_TRAIN_ITERS):
        
        # 取得 "目前" Policy (pi) 對 (S, A) 的 log probabilities
        log_probs_new = torch.nn.LogSoftmax(dim=-1)(pi(S)).gather(1, A.view(-1, 1)).view(-1)
        
        # 計算 new_probs / old_probs 的比例 (Ratio)
        # r_t = exp(log_probs_new - old_log_probs)
        ratio = torch.exp(log_probs_new - old_log_probs)

        # 計算 PPO-Clip Objective 的兩個部分
        # 1. 未裁剪 (Unclipped) 的 surrogate objective
        surr1 = ratio * A_t
        
        # 2. 裁剪後 (Clipped) 的 surrogate objective
        # torch.clamp 將 ratio 限制在 [1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM] 之間
        surr2 = torch.clamp(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * A_t

        # PPO 的 Loss 是兩者的最小值，再取平均
        # 我們要 "最大化" (maximize) 這個 objective，
        # 但 optimizer 只能 "最小化" (minimize)，所以我們在前面加上負號
        policy_loss = -torch.min(surr1, surr2).mean()

        # 更新 Policy Network
        OPT2.zero_grad()
        policy_loss.backward()
        OPT2.step()
        
    #################################

# Play episodes
Rs = [] 
last25Rs = []
print("Training:")
pbar = tqdm.trange(EPOCHS)
for epi in pbar:

    all_S, all_A = [], []
    all_returns = []
    for epj in range(EPISODES_PER_EPOCH):
        
        # Play an episode and log episodic reward
        S, A, R = utils.envs.play_episode(env, policy)
        
        # modify the reward for "mountain_car_mod" mode
        # replace reward with the height of the car (which is first component of state)
        if args.mode == "mountain_car_mod":
            R = [s[0] for s in S[:-1]]
            

        all_S += S[:-1] # ignore last state
        all_A += A
        
        # Create returns 
        discounted_rewards = copy.deepcopy(R)
        for i in range(len(R)-1)[::-1]:
            discounted_rewards[i] += GAMMA * discounted_rewards[i+1]
        discounted_rewards = t.f(discounted_rewards)
        all_returns += [discounted_rewards]

    Rs += [sum(R)]
    S, A = t.f(np.array(all_S)), t.l(np.array(all_A))
    returns = torch.cat(all_returns, dim=0).flatten()

    # pi(a |s) for trajectories observed above
    log_probs = torch.nn.LogSoftmax(dim=-1)(pi(S)).gather(1, A.view(-1, 1)).view(-1)

    # train
    # .detach() 是為了將 old_log_probs 視為常數，在計算 PPO ratio 時不會追蹤它的梯度
    train(S, A, returns, log_probs.detach())

    # Show mean episodic reward over last 25 episodes
    last25Rs += [sum(Rs[-25:])/len(Rs[-25:])]
    pbar.set_description("R25(%g, mean over 10 episodes)" % (last25Rs[-1]))
  
pbar.close()
print("Training finished!")

# Plot the reward
N = len(last25Rs)
plt.plot(range(N), last25Rs, 'b')
plt.xlabel('Episode')
plt.ylabel('Reward (averaged over last 25 episodes)')
plt.title("PPO, mode: " + args.mode)
plt.savefig("images/ppo-"+args.mode+".png")
print("Episodic reward plot saved!")

# Play test episodes
print("Testing:")
testRs = []
for epi in range(TEST_EPISODES):
    S, A, R = utils.envs.play_episode(env, policy, render = False)
    
    # modify the reward for "mountain_car_mod" mode
    # replace reward with the height of the car (which is first component of state)
    if "mountain_car" in args.mode:
        R = [s[0] for s in S[:-1]]

    testRs += [sum(R)]
    print("Episode%02d: R = %g" % (epi+1, sum(R)))

if "mountain_car" in args.mode:
    print("Height achieved: %.2f ± %.2f" % (np.mean(testRs), np.std(testRs)))
else:
    print("Eval score: %.2f ± %.2f" % (np.mean(testRs), np.std(testRs)))
env.close()