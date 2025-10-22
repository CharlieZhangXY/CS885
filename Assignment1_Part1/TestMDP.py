from MDP import *
import numpy as np

''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
# Transition function: |A| x |S| x |S'| array
T = np.array([
    [[0.5, 0.5, 0, 0], [0, 1, 0, 0], [0.5, 0.5, 0, 0], [0, 1, 0, 0]],
    [[1, 0, 0, 0], [0.5, 0, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0, 0.5, 0.5]]
])

# Reward function: |A| x |S| array
R = np.array([[0, 0, 10, 10],
              [0, 0, 10, 10]])

# Discount factor: scalar in [0,1)
discount = 0.9        

# MDP object
mdp = MDP(T, R, discount)

'''Test each procedure'''
# Value Iteration
[V, nIterations, epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates))
print("=== Value Iteration ===")
print("V =", V)
print("Iterations =", nIterations)
print("Epsilon =", epsilon)

# Extract Policy
policy = mdp.extractPolicy(V)
print("\n=== Extracted Policy from Value Iteration ===")
print("Policy =", policy)

# Policy Evaluation
V = mdp.evaluatePolicy(np.array([1, 0, 1, 0]))
print("\n=== Policy Evaluation (given [1,0,1,0]) ===")
print("V =", V)

# Policy Iteration
[policy, V, iterId] = mdp.policyIteration(np.array([0, 0, 0, 0]))
print("\n=== Policy Iteration ===")
print("Policy =", policy)
print("V =", V)
print("Iterations =", iterId)

# Partial Policy Evaluation
[V, iterId, epsilon] = mdp.evaluatePolicyPartially(np.array([1, 0, 1, 0]), np.array([0, 10, 0, 13]))
print("\n=== Partial Policy Evaluation ===")
print("V =", V)
print("Iterations =", iterId)
print("Epsilon =", epsilon)

# Modified Policy Iteration
[policy, V, iterId, tolerance] = mdp.modifiedPolicyIteration(np.array([1, 0, 1, 0]), np.array([0, 10, 0, 13]))
print("\n=== Modified Policy Iteration ===")
print("Policy =", policy)
print("V =", V)
print("Iterations =", iterId)
print("Tolerance =", tolerance)
