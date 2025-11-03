import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        V = initialV.copy()
        iterId = 0
        epsilon = float('inf')
        
        while iterId < nIterations and epsilon > tolerance:
            V_old = V.copy()
            
            # 對每個狀態計算 max_a [R^a + gamma * T^a * V]
            Q = np.zeros((self.nActions, self.nStates))
            for a in range(self.nActions):
                Q[a] = self.R[a] + self.discount * np.dot(self.T[a], V)
            
            V = np.max(Q, axis=0)
            
            epsilon = np.max(np.abs(V - V_old))
            iterId += 1
        
        return [V, iterId, epsilon]

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        Q = np.zeros((self.nActions, self.nStates))
        for a in range(self.nActions):
            Q[a] = self.R[a] + self.discount * np.dot(self.T[a], V)
        
        policy = np.argmax(Q, axis=0)

        return policy 

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        # 構建 R^pi 和 T^pi
        R_pi = np.zeros(self.nStates)
        T_pi = np.zeros((self.nStates, self.nStates))
        
        for s in range(self.nStates):
            a = int(policy[s])
            R_pi[s] = self.R[a, s]
            T_pi[s] = self.T[a, s]
        
        # 解線性方程組: V = R^pi + gamma * T^pi * V
        # (I - gamma * T^pi) * V = R^pi
        A = np.eye(self.nStates) - self.discount * T_pi
        V = np.linalg.solve(A, R_pi)

        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        policy = initialPolicy.copy()
        iterId = 0
        
        while iterId < nIterations:
            # Policy evaluation
            V = self.evaluatePolicy(policy)
            
            # Policy improvement
            new_policy = self.extractPolicy(V)
            
            # 檢查是否收斂
            if np.array_equal(policy, new_policy):
                break
            
            policy = new_policy
            iterId += 1

        return [policy, V, iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        V = initialV.copy()
        iterId = 0
        epsilon = float('inf')
        
        # 構建 R^pi 和 T^pi
        R_pi = np.zeros(self.nStates)
        T_pi = np.zeros((self.nStates, self.nStates))
        
        for s in range(self.nStates):
            a = int(policy[s])
            R_pi[s] = self.R[a, s]
            T_pi[s] = self.T[a, s]
        
        while iterId < nIterations and epsilon > tolerance:
            V_old = V.copy()
            V = R_pi + self.discount * np.dot(T_pi, V)
            epsilon = np.max(np.abs(V - V_old))
            iterId += 1

        return [V, iterId, epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        policy = initialPolicy.copy()
        V = initialV.copy()
        iterId = 0
        epsilon = float('inf')
        
        while iterId < nIterations and epsilon > tolerance:
            V_old = V.copy()
            
            # Partial policy evaluation
            [V, _, _] = self.evaluatePolicyPartially(policy, V, nEvalIterations, tolerance)
            
            # Policy improvement
            policy = self.extractPolicy(V)
            
            epsilon = np.max(np.abs(V - V_old))
            iterId += 1

        return [policy, V, iterId, epsilon]