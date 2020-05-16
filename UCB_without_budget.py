def fairness_compute_UCB(H, r, t, alpha=0):
  return np.max(np.maximum(t*r - H,0))

def regret_compute_UCB(t,r,u,S_hat):
  reward_received = np.sum(S_hat)
  reward_optimal_fair = np.sum(t*r*u)+(1-np.sum(r))*t*np.max(np.array(u))
  return reward_optimal_fair - reward_received

def UCB(numbers_of_selections,sums_of_reward,n):
    Arm = 0
    max_upper_bound = 0
    for i in range(0, len(numbers_of_selections)):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_reward[i] / numbers_of_selections[i]
            delta_i = math.sqrt(2 * math.log(n) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            Arm = i
    return Arm

def Main_Algo(k, T, r, u, alpha):
  N = np.array([1]*k)
  S = []  #Total reward
  RR = []
  FV = []
  for f in range(k):
      S.append(bernoulli.rvs(u[f], size=1)[0])
  # S = [0]*k
  for t in range(k+1,T+1):
      temp = []
      A = []
      A_Val = []
      for i in range(k):
          if (r[i]*(t-1) - N[i]) > alpha:
              A.append(i)
              A_Val.append(r[i]*(t-1) - N[i])
      if A:
          Arm = A[np.argmax(np.array(A_Val))]
      else:
          Arm = UCB(N,S,t)
      N[Arm] += 1
      S[Arm] += bernoulli.rvs(u[Arm], size=1)[0]
      RR.append(regret_compute_UCB(t,r,u,S))
      #print(fairness_compute_UCB(N, r, t))
      FV.append(fairness_compute_UCB(N, r, t, alpha))
  return N, S, np.array(RR), np.array(FV)


if __name__ == "__main__":
