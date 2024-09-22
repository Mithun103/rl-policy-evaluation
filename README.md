# POLICY EVALUATION

## AIM
The aim is to implement a reinforcement learning agent that can navigate environments from the gym-walk library, which simulates grid-like or walking environments. The agent will learn an optimal policy to reach desired goals based on the given reward structure.

## PROBLEM STATEMENT
The task is to implement and evaluate a reinforcement learning agent in a walking environment using gym. The agent must learn to make decisions that maximize its total reward through trial and error, based on feedback from the environment.

## POLICY EVALUATION FUNCTION
The policy evaluation function aims to compute the value of a given policy by iteratively calculating the expected rewards of following the policy in each state until convergence, allowing for better estimation of state values under the current policy.
## Program:
```
import warnings ; warnings.filterwarnings('ignore')

import gym, gym_walk
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123)

# Reference https://github.com/mimoralea/gym-walk

def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)

def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)

# Slippery Walk Five MDP



env = gym.make('SlipperyWalkFive-v0')
P = env.env.P
init_state = env.reset()
goal_state = 6
LEFT, RIGHT = range(2)

P

init_state

state, reward, done, info = env.step(RIGHT)
print("state:{0} - reward:{1} - done:{2} - info:{3}".format(state, reward, done, info))

# First Policy
pi_1 = lambda s: {
    0:LEFT, 1:LEFT, 2:LEFT, 3:LEFT, 4:LEFT, 5:LEFT, 6:LEFT
}[s]
print_policy(pi_1, P, action_symbols=('<', '>'), n_cols=7)

# Find the probability of success and the mean return of the first policy
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_1, goal_state=goal_state)*100,
    mean_return(env, pi_1)))

# Create your own policy
pi_2= lambda s: {
    0:RIGHT, 1:RIGHT, 2:RIGHT, 3:LEFT, 4:LEFT, 5:LEFT, 6:LEFT
}[s]
# Write your code here

print_policy(pi_2, P, action_symbols=('<', '>'), n_cols=7)

# Find the probability of success and the mean return of you your policy

#write your code here

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_2, goal_state=goal_state)*100,
    mean_return(env, pi_2)))

# Compare your policy with the first policy
print("first policy getting a undiscounted return of 0.0300 Which reaches goal 3.00% \nSecond policy getting a undiscounted resturn of 0.1800\nwhich is greater than firtpolicy\nso comparatively second policy is better than first policy")

# Policy Evaluation

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    prev_V = np.zeros(len(P))
    while True:
        V = np.zeros(len(P))
        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
        if np.max(np.abs(prev_V - V)) < theta:
            break
        prev_V = V.copy()

    return V

# Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P)
print_state_value_function(V1, P, n_cols=7, prec=5)

# Code to evaluate the second policy
V2= policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)


# Compare the two policies based on the value function using the above equation and find the best policy

V1

print_state_value_function(V1, P, n_cols=7, prec=5)

V2

print_state_value_function(V2, P, n_cols=7, prec=5)

V1>=V2

print("Name : MITHUN M S             Regno: 212222240067")

if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better poicy")
else:
  print("Both policies have their merits.")
  ```
## OUTPUT:
First policy:
![image](https://github.com/user-attachments/assets/476b448e-656e-47ff-8b2a-ffa6cb99f919)
Second Policy:
![image](https://github.com/user-attachments/assets/5f2eb13d-0980-458a-90a8-d6f5223a6acf)
Best Policy:
![image](https://github.com/user-attachments/assets/4dcf4662-9dae-42aa-b71a-f1a37ba69a24)




## RESULT:

Thus, the reinforcement learning agent successfully learns an optimal policy for navigating the environment, improving its decisions through iterations.
