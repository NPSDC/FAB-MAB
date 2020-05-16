from budget_helper import initialize, update, compute_best_arm
import numpy as np
from scipy.stats import bernoulli
from random import seed
from random import random

def initialize_dist(K):
    beta_distn_reward = []
    beta_distn_cost = []

    for i in range(K):
        beta_distn_reward.append([0.0,0.0])

    for i in range(K):
        beta_distn_cost.append([0.0,0.0])

    return beta_distn_reward, beta_distn_cost

def sample_reward_and_cost(arm_pulled, mu, cost):
    reward_received = bernoulli.rvs(mu[arm_pulled], size=1)[0]
    cost_received = bernoulli.rvs(cost[arm_pulled], size=1)[0]
    return reward_received, cost_received

def update_distn_and_budget(arm_pulled, budget, beta_distn_reward, beta_distn_cost, reward_received, cost_received):
    # cost_records.append(cost_received)
    # reward_records.append(reward_received)
    # arm_pulled_records.append(arm_pulled)
    beta_distn_reward[arm_pulled][0] += reward_received
    beta_distn_reward[arm_pulled][1] += (1-reward_received)
    beta_distn_cost[arm_pulled][0] += cost_received
    beta_distn_cost[arm_pulled][1] += (1-cost_received)
    return beta_distn_reward, beta_distn_cost

def choose_arm(beta_distn_reward, beta_distn_cost, k, budget, mu, cost):
    # reward_records = []
    # arm_pulled_records = []
    # cost_records = []

    sampled_mean_reward = np.array([0]*k, dtype=np.float)
    sampled_mean_cost = np.array([0]*k, dtype=np.float)

    if budget>0:
        for arm in range(k):
            sampled_mean_reward[arm] = np.random.beta(beta_distn_reward[arm][0]+1, beta_distn_reward[arm][1]+1)
            sampled_mean_cost[arm] = np.random.beta(beta_distn_cost[arm][0]+1, beta_distn_cost[arm][1]+1)
        arm_pulled = np.argmax(sampled_mean_reward/sampled_mean_cost)
        #reward_received, cost_received = sample_reward_and_cost(arm_pulled, mu, cost)
        #budget, beta_distn_reward, beta_distn_cost = update_distn_and_budget(arm_pulled, budget, beta_distn_reward, beta_distn_cost, reward_received,cost_received)
    return arm_pulled
    #return reward_received, cost_received, arm_pulled, budget, beta_distn_reward, beta_distn_cost

def fairness_with_budget_thompson_sampling(K, budget, mu, cost, alpha = None, R = None):
    
    min_mu_cost = min(cost)
    arm_pulled_count, reward_records, cost_records, arm_pulled_records = initialize(K, budget, min_mu_cost)

    t = 0
    beta_distn_reward, beta_distn_cost = initialize_dist(K)

    while budget > 0:
        t += 1
        play_thomp = True
        arm_pulled = -1
        if(alpha is not None and R is not None):
            unfair_arm = []
            unfair_val = []
            
            for i in range(K):
                if (R[i]*(t-1) - arm_pulled_count[t][i]) > alpha:
                    unfair_arm.append(i)
                    unfair_val.append(R[i]*(t-1) - arm_pulled_count[t][i])

            if unfair_arm:
                play_thomp = False
                arm_pulled = unfair_arm[np.argmax(np.array(unfair_val))]
                
        if play_thomp:
            arm_pulled = choose_arm(beta_distn_reward, beta_distn_cost, K, budget, mu, cost)
        
        update(arm_pulled_count, reward_records, cost_records, t)
        
        reward_received, cost_received = sample_reward_and_cost(arm_pulled, mu, cost)        
        beta_distn_reward, beta_distn_cost = update_distn_and_budget(arm_pulled, budget, beta_distn_reward, 
                                   beta_distn_cost, reward_received, cost_received)
        
        arm_pulled_count[t][arm_pulled] += 1
        budget -= cost_received
        cost_records[t][arm_pulled] += cost_received
        reward_records[t][arm_pulled] += reward_received
        arm_pulled_records[t] = arm_pulled

    return arm_pulled_count, reward_records, cost_records, arm_pulled_records, t #N,