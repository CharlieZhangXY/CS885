import numpy as np
import MDP

''' Construct a simple maze MDP'''
# [省略 T 矩陣定義以節省空間 - 使用原始 TestMDPmaze.py 中的定義]

# Transition function: |A| x |S| x |S'| array
T = np.zeros([4,17,17])
a = 0.8;  b = 0.1;

# [這裡包含完整的 T 矩陣定義 - 與原始檔案相同]
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

# Reward function
R = -1 * np.ones([4,17])
R[:,14] = 100  # goal state
R[:,9] = -70   # bad state
R[:,16] = 0    # end state

discount = 0.95
mdp = MDP.MDP(T,R,discount)

# ==================== 問題 1: Value Iteration ====================
print("=" * 60)
print("問題 1: Value Iteration")
print("=" * 60)

[V, nIterations, epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates), tolerance=0.01)
policy_vi = mdp.extractPolicy(V)

print(f"\n迭代次數: {nIterations}")
print(f"最終 epsilon: {epsilon:.6f}\n")

print("策略 (Policy):")
action_names = ['上(up)', '下(down)', '左(left)', '右(right)']
for s in range(mdp.nStates):
    print(f"  State {s:2d}: Action {int(policy_vi[s])} ({action_names[int(policy_vi[s])]})")

print("\n值函數 (Value Function):")
for s in range(mdp.nStates):
    print(f"  V[{s:2d}] = {V[s]:7.2f}")

# ==================== 問題 2: Policy Iteration ====================
print("\n" + "=" * 60)
print("問題 2: Policy Iteration")
print("=" * 60)

[policy_pi, V_pi, iterId_pi] = mdp.policyIteration(np.zeros(mdp.nStates, dtype=int))

print(f"\n迭代次數: {iterId_pi}")

print("\n策略 (Policy):")
for s in range(mdp.nStates):
    print(f"  State {s:2d}: Action {int(policy_pi[s])} ({action_names[int(policy_pi[s])]})")

print("\n值函數 (Value Function):")
for s in range(mdp.nStates):
    print(f"  V[{s:2d}] = {V_pi[s]:7.2f}")

# ==================== 問題 3: Modified Policy Iteration ====================
print("\n" + "=" * 60)
print("問題 3: Modified Policy Iteration")
print("=" * 60)

print("\n測試不同的 nEvalIterations (1-10):\n")
print("nEvalIterations | 總迭代次數")
print("-" * 35)

results = []
for nEval in range(1, 11):
    [policy_mpi, V_mpi, iterId_mpi, epsilon_mpi] = mdp.modifiedPolicyIteration(
        np.zeros(mdp.nStates, dtype=int),
        np.zeros(mdp.nStates),
        nEvalIterations=nEval,
        tolerance=0.01
    )
    results.append((nEval, iterId_mpi))
    print(f"      {nEval:2d}        |     {iterId_mpi:3d}")

print("\n分析:")
print(f"  - nEvalIterations = 1:  {results[0][1]} 次迭代 (類似 value iteration)")
print(f"  - nEvalIterations = 10: {results[9][1]} 次迭代 (接近 policy iteration)")
print(f"  - Policy iteration:     {iterId_pi} 次迭代")
print(f"  - Value iteration:      {nIterations} 次迭代")

print("\n結論:")
print("  Modified policy iteration 在 value iteration 和 policy iteration 之間")
print("  提供了靈活的折衷選擇。較大的 nEvalIterations 能更快收斂，但")
print("  每次迭代的計算成本更高。")

print("\n" + "=" * 60)