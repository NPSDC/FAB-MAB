from ucb_budget import UCB_with_budget_and_fair
from thompson_budget import fairness_with_budget_thompson_sampling
from budget_helper import *
import pickle as pi
import os
import sys

def run_mab(k, B, nTimes, reward_means, cost_means, method = "UCB", alpha = None, R = None, pickle_file = None):
    #other_vars = {"N":[], "X":[], "C":[], "arm_pulled":[], "tB":[], "bud_arr":[], "reg_arr":[]}
    other_vars = {"bud_arr":[], "reg_arr":[]}
    if not alpha is None:
        other_vars["FV"] = [] 
    for i in range(nTimes):
        print("iteration ", i)
        if(method == "UCB"):
            N, X, C, arm_pulled, tB = UCB_with_budget_and_fair(k, B, reward_means, cost_means, alpha = alpha, R = R)
        elif(method == "thompson"):
            N, X, C, arm_pulled, tB = fairness_with_budget_thompson_sampling(k, B, reward_means, cost_means, alpha = alpha, R = R)
        else:
            sys.exit("Invalid input")
        if not alpha is None:
            FV = np.array(range(0, tB + 1), ndmin = 2).T @ np.array(R, ndmin = 2) - N[0:tB+1, ]
            FV[FV < 0] = 0
            other_vars["FV"].append(np.max(FV, axis = 1))

        # other_vars["N"].append(N)
        # other_vars["X"].append(X)
        # other_vars["C"].append(C)
        # other_vars["arm_pulled"].append(arm_pulled)
        # other_vars["tB"].append(tB)
        regret, reg_sum, budget_sum = compute_regret(C, arm_pulled, cost_means, reward_means, tB)
        other_vars["bud_arr"].append(budget_sum)
        other_vars["reg_arr"].append(reg_sum)

    if pickle_file is not None:
        with open(pickle_file, "wb") as w:
            pi.dump(other_vars, w)
    if(not alpha is None):
        return other_vars["bud_arr"], other_vars["reg_arr"], other_vars["FV"]
    return other_vars["bud_arr"], other_vars["reg_arr"]

def compute_average(regret_array, budget_array, fair_viol_array = None, interval = 500, pick_file = None):
    """Parameters
    regret_array - A list of numpy arrays of size N*tB containing regret, N is the number of times an experiment has been run, tB is varying the time required to reacha budget
    regret_array - A list of numpy arrays of size N*tB containing budget, N is the number of times an experiment has been run, tB is varying the time required to reacha budget
    """
    match = lambda a, b: np.array([ np.where(b == x)[0][0] if x in b else None for x in a ])
    n_trials = len(regret_array)
    min_budget = 1e10
    for i in range(len(budget_array)):
        min_budget = min(budget_array[i][-1], min_budget)
    
    #print(min_budget)
    time_bud = range(0, int(min_budget) + 1, interval)
    inds_match = list()
    
    if(len(regret_array) != len(budget_array)):
        sys.exit("number of dimensions not same")
    if(np.sum(regret_array[0].shape == budget_array[0].shape) != len(regret_array[0].shape)):
        sys.exit("Regret shape not same as budget shape")
    
    av_reg = np.zeros(len(time_bud))
    for i in range(n_trials):
        inds_match.append(match(time_bud, budget_array[i]))
        av_reg += regret_array[i][inds_match[i]]   
    av_reg /= n_trials

    if(fair_viol_array is not None):
        av_fair_viol = np.zeros(len(time_bud))
        for i in range(n_trials):
            av_fair_viol += fair_viol_array[i][inds_match[i]]
        av_fair_viol /= n_trials

    if(pick_file is not None):
        comb = (av_reg)
        if(fair_viol_array is not None):
            comb = (av_reg, av_fair_viol)
        pi.dump(comb, open(pick_file, "wb"))

    if(fair_viol_array is not None):
        return inds_match, av_reg, av_fair_viol

    return inds_match, av_reg

def run_alphas_mab(k, B, nTimes, reward_means, cost_means, method, alphas, R, pick_file = None):
    reg_alpha = []
    fair_viol_alpha = []
    if method in ["thompson", "UCB"]:
        for alpha in alphas:
            print("alpha ", alpha)
            bud_arr, reg_array, fair_viol_array = run_mab(k, B, nTimes, reward_means, cost_means, method, alpha, R, None)   
            inds, reg, fair_viol = compute_average(reg_array, bud_arr, fair_viol_array, interval = 100)
            #print(reg)
            #print(fair_viol)
            reg_alpha.append(reg[-1])
            fair_viol_alpha.append(fair_viol[-1])
    else:
        sys.exit("Invalid input")

    if(pick_file is not None):
        comb = (reg_alpha, fair_viol_alpha)
        pi.dump(comb, open(pick_file, "wb"))

    return reg_alpha, fair_viol_alpha

if __name__ == "__main__":
    nTimes = 10 ## number of times to run the algorithm
    B = 1000
    k = 10
    reward_means = np.random.rand(10)
    cost_means = np.random.choice(range(1,k*10+1), size = k, replace = False)/10
    bud_arr, reg_array = run_mab(k, B, 10, reward_means, cost_means, "ucb_bud.pickle")
    inds, reg = compute_regret(reg_array, bud_arr, interval = 100)

