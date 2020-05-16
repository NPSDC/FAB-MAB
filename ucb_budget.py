from budget_helper import initialize, update, compute_best_arm
import numpy as np
from scipy.stats import bernoulli
from random import seed
from random import random

# neg_array = [-10]*10
# temp_arm_pulled = np.array([0]*10)
explore_max = 0
exploit_max = 0

def UCB_BV2(X, C, N, lam, t):
    #D_t = X[t]/C[t] + (1 + 1/(lam - np.sqrt(np.log(t)/N[t])))*np.sqrt(np.log(t)/N[t])*1/lam
	av_rew = X[t]/N[t]
	av_cost = C[t]/N[t]
	av_cost[(av_cost - 0) <= 1e-5] = 1e-3
	exploit = av_rew/av_cost
	sq_term = np.sqrt(np.log(t+1)/N[t])
	explore = (1 + 1/(lam - sq_term))*sq_term*1/lam

	global neg_array
	global explore_max
	global exploit_max
	exploit_max = max(exploit_max, max(exploit))
	explore_max = max(explore_max, max(explore))
	neg_inds = explore<0 
	# if(sum(neg_inds) > 0):


		# print([t*sum(neg_inds)])
		# print(neg_inds)
		# print(neg_array[neg_inds])
		# neg_array[neg_inds] = np.minimum([t*sum(neg_inds)], neg_array[neg_inds])
	#print(exploit)
	#print(explore)
	#if(t > 1000 and )
	D_t = exploit + explore
	# for i in range(len(explore)):
	# 	if D_t[i]<0:
	# 		#print(neg_array[i])
	# 		neg_array[i] = max(t, int(neg_array[i]))
	return np.argmax(D_t)
    
def UCB_with_budget_and_fair(K, B, reward_means, cost_means, alpha = None, R = None):
	"""
	Parameters
	K - Number of arms
	B - Budget
	reward_means - An array bernoulli means for rewards
	cost_means - An array bernoulli means for costs
	"""
	min_mu_cost = min(cost_means)
	N, X, C, arm_pulled = initialize(K, B, min_mu_cost)
	#globaltemp_arm_pulled
	##Playing each arm atleast once
	for arm in range(K):
		if arm > 0:
			update(N, X, C, arm)
		N[arm][arm] += 1
		X[arm][arm] += bernoulli.rvs(reward_means[arm], size=1)[0]
		C[arm][arm] += bernoulli.rvs(cost_means[arm], size=1)[0]
		arm_pulled[arm] = arm
		#temp_arm_pulled[arm] = arm
	t = K-1
	#print(B)
	B -= np.sum(C[t])
	#print(B)
	count=0
	while B > 0:
		t = t + 1
		ave_costs = C[t-1]/N[t-1]
		lam = np.min(ave_costs) ##Hack
		lam = lam if(lam - 0) > 1e-5 else 0.001
	    #print(lam)
	    ###Fairness
		fair_penalty = np.zeros(K)
		if(alpha is not None and R is not None):

			fair_penalty = R*t - N[t-1] #T is the actual time step, its index represented by t-1
			if(np.max(fair_penalty) > alpha):
				arm = np.argmax(fair_penalty)
			else:
				arm = UCB_BV2(X, C, N, lam, t-1)
		else:        
			arm = UCB_BV2(X, C, N, lam, t-1)
		arm_pulled[t] = arm
		#temp_arm_pulled[arm] = t
	    #print(t)
		update(N, X, C, t)
		N[t][arm] += 1
		X[t][arm] += bernoulli.rvs(reward_means[arm], size=1)[0]
		cost = bernoulli.rvs(cost_means[arm], size=1)[0]
		C[t][arm] += cost
		#print(arm)
		# if arm==5:
		# 	print("as")
		# 	#print(cost_means[arm])
		# 	if cost != 0:
		# 		count+=1
		B -= cost
	#print (count)
	return N,X,C,arm_pulled,t

from collections import Counter
def main():
	nTimes = 10 ## number of times to run the algorithm
	B = 1000
	k = 4
	nTimes = 1
	reward_means = [0]*4
	reward_means[3] = 1
	cost_means = [0.1]*4
	cost_means[3] = 0.1
	N, X, C, arm_pulled, tB = UCB_with_budget_and_fair(k, B, reward_means, cost_means)
	print(Counter(arm_pulled[0:tB+1]))
	print(explore_max, exploit_max)
	#print(temp_arm_pulled, neg_array)

if __name__ == "__main__":
	main()
	