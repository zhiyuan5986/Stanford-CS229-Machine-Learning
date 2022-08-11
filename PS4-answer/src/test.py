from __future__ import division, print_function
from env import CartPole, Physics
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter

NUM_STATES = 163
num_states = NUM_STATES

transition_counts = np.zeros((num_states, num_states, 2))
transition_probs = np.ones((num_states, num_states, 2)) / num_states
    # Index zero is count of rewards being -1 , index 1 is count of total num state is reached
reward_counts = np.zeros((num_states, 2))
reward = np.zeros(num_states)
value = np.random.rand(num_states) * 0.1

mdp_data = {
        'transition_counts': transition_counts,
        'transition_probs': transition_probs,
        'reward_counts': reward_counts,
        'reward': reward,
        'value': value,
        'num_states': num_states,
    }

state = 0
print(transition_probs[state].T.shape)
print(value.shape)
pi = transition_probs[state].T @ value
print(np.argmax(pi))

print(reward_counts[:,1] != 0)
print(transition_counts[1:5][:,:,1].shape)
prob = transition_counts[1:5][:,:,1] / transition_counts[1:5][:,:,0]
print(transition_counts[1:5][:,:,0].shape)

print(transition_probs.shape)
print(transition_probs[:,:,0] @ value)
value_old = value
print(mdp_data['reward'].shape)
print(mdp_data['reward']+np.max(np.vstack((mdp_data['transition_probs'][:,:,0]@value_old,mdp_data['transition_probs'][:,:,1]@value_old)),axis=0))

print(transition_counts[reward_counts[:,1] == 0][:,:,0].shape)