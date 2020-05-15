import numpy as np
from scipy.stats import bernoulli
from random import seed
from random import random

def initialize(K, B, min_mu_cost):
    """
    Parameters
    K - Number of arms
    B - Budget
    min_mu_cost - The minimum cost in the bernoulli trial
    Returns
    max_size - 2*B/min_cost
    N - A 2d array of size max_size and arms containing the total number of times an arm has been played at each time step
    X - A 2d array of size max_size and arms containing total reward obtained at each time step
    C - A 2d array of size max_size and arms containing total cost obtained at each time step
    arm_
    """
    max_size = int(2*B/min_mu_cost)
    N = np.zeros((max_size, K))
    X = np.zeros((max_size, K))
    C = np.zeros((max_size, K))
    arm_pulled = np.zeros(max_size, dtype = "int")
    return N,X,C,arm_pulled


def update(N, X, C, t):
    N[t] = N[t-1]
    X[t] = X[t-1]
    C[t] = C[t-1]

def compute_regret(C, arm_pulled, cost_means, reward_means, tB):
    """
    Returns 
    regret - an array size of tB+1 containing regret at each round
    #budget - total budget at end of round t
    """
    regret = np.zeros(tB+1)
    reg_sum = np.zeros(tB+1)
    budget_sum = np.zeros(tB+1)
    best_arm = compute_best_arm(reward_means, cost_means)
    #budget = np.zeros(tB+1)
    for i in range(tB+1):
        #budget[i] = np.sum(C[t])
        arm = arm_pulled[i]
        regret[i] = cost_means[arm]*(reward_means[best_arm]/cost_means[best_arm] - reward_means[arm]/cost_means[arm])
        #regret[i] = reward_means[best_arm] - reward_means[arm]
        reg_sum[i] = (reg_sum[i-1] + regret[i]) if i > 0 else regret[i]
        budget_sum[i] = np.sum(C[i])
        
    return regret, reg_sum, budget_sum

def compute_best_arm(mu, cost):
    return np.argmax(np.array(mu)/np.array(cost))
