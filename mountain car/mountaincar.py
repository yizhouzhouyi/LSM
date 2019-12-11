# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:57:11 2018

@author: yizhou
"""

import matplotlib
import math
import numpy as np
import pandas as pd
import gym
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from mpl_toolkits.mplot3d import Axes3D
import sklearn.pipeline
import sklearn.preprocessing
from scipy.linalg import block_diag, eigh, svd
from scipy.sparse.csgraph import laplacian
from car_environment import MountainCarEnv

#env = gym.make('MountainCar-v0')
env = MountainCarEnv()
num_episodes = 50
discount_factor = 1.0
alpha = 0.01
nA = env.action_space.n#3
#Parameter vector define number of parameters per action based on featurizer size
w = np.zeros((nA,400))

#w1 = np.load('1.npy')
w3 = np.load('3.npy')
w4 = np.load('4-2.npy')
#source_w1 = w1.reshape(1,-1)
source_w3 = w3.reshape(1,-1)
source_w4 = w4.reshape(1,-1)
source_w = np.vstack((source_w3,source_w4))

# Plots
plt_actions = np.zeros(nA)#[0 0 0]
#episode_rewards = np.zeros(num_episodes)#[0 0 ...0]200

# Get satistics over observation space samples for normalization
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
#print(observation_examples[0])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Create radial basis function sampler to convert states to features for nonlinear function approx
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
		])

# Fit featurizer to our scaled inputs
featurizer.fit(scaler.transform(observation_examples))

def cos_sim(a,b):
    a = np.mat(a) 
    b = np.mat(b) 
    num = float(a * b.T) 
    denom = np.linalg.norm(a) * np.linalg.norm(b) 
    cos = num / denom 
    sim = 0.5 + 0.5 * cos 
    return sim

def compute_cxy(source_q,task_q):
    cxy = np.zeros((source_q.shape[0],task_q.shape[0]))
    for i in range(source_q.shape[0]):
        sim = cos_sim(source_q[i],task_q)
        if sim > 0.70:
            cxy[i,:] = 1
        else:
            cxy[i,:] = 0
    return cxy

def low_rank_align(X, Y, Cxy, d, mu=0.8):
    nx, dx = X.shape  #X的大小
    ny, dy = Y.shape  #Y的大小
    #assert Cxy.shape==(nx,ny), \
    C = np.fliplr(block_diag(np.fliplr(Cxy),np.fliplr(Cxy.T)))  #C
    #if d is None:
        #d = min(dx,dy)
    Rx = low_rank_repr(X,d)
    Ry = low_rank_repr(Y,d)
    R = block_diag(Rx,Ry)  #R
    tmp = np.eye(R.shape[0]) - R
    M = tmp.T.dot(tmp)
    L = laplacian(C)
    eigen_prob = (1-mu)*M + 2*mu*L
    _,F = eigh(eigen_prob,eigvals=(1,d),overwrite_a=True,overwrite_b=True)#eigvals=(1,d),overwrite_a=True
    Xembed = F[:nx]
    Yembed = F[nx:]
    return Xembed, Yembed

def low_rank_repr(X, n_dim):
    U, S, V = svd(X.T,full_matrices=False)
    mask = S > 1
    V = V[mask]
    S = S[mask]
    R = (V.T * (1 - S**-2)).dot(V)
    return R
                
# Normalize and turn into feature
def featurize_state(state):
	# Transform data
	scaled = scaler.transform([state])
	featurized = featurizer.transform(scaled)
	return featurized

def Q(state,action,w):
	value = state.dot(w[action])
	return value

# Epsilon greedy policy
def policy(state, weight, epsilon=0.9):
    if np.random.uniform()<epsilon:
        action = np.argmax([Q(state,a,w) for a in range(nA)])
    else:
        action = np.random.choice(nA)
    return action

def ma_policy(state, weight, epsilon=0.9):
    if np.random.uniform()<epsilon:
        action = np.argmax([Q(state,a,w) for a in range(nA)])
    else:
        target_w = weight.reshape(1,-1)
        Cxy = compute_cxy(source_w,target_w)
        Xembed,Yembed = low_rank_align(source_w,target_w,Cxy,2)
        similarity = []
        for i in range(Xembed.shape[0]):
            similarity.append(cos_sim(Xembed[i],Yembed))
        best_source = np.argmax(similarity)
        best_source_w = source_w[best_source]
        best_source_w = best_source_w.reshape(3,400)
        action = np.argmax([Q(state,a,best_source_w) for a in range(nA)])
    return action

def entropy_policy(state, weight, epsilon=0.9):
    q_sa = [Q(state,a,w) for a in range(nA)]
    qs_max = np.max(q_sa)
    psa0 = math.exp(q_sa[0]-qs_max)/(math.exp(q_sa[0]-qs_max)+math.exp(q_sa[1]-qs_max)+math.exp(q_sa[2]-qs_max))
    psa1 = math.exp(q_sa[1]-qs_max)/(math.exp(q_sa[0]-qs_max)+math.exp(q_sa[1]-qs_max)+math.exp(q_sa[2]-qs_max))
    psa2 = math.exp(q_sa[2]-qs_max)/(math.exp(q_sa[0]-qs_max)+math.exp(q_sa[1]-qs_max)+math.exp(q_sa[2]-qs_max))
    Hs = -(psa0*math.log(psa0,3)+psa1*math.log(psa1,3)+psa2*math.log(psa2,3))
    epsilon = Hs
    if np.random.uniform()<epsilon:
        action = np.argmax([Q(state,a,w) for a in range(nA)])
    else:
        target_w = weight.reshape(1,-1)
        Cxy = compute_cxy(source_w,target_w)
        Xembed,Yembed = low_rank_align(source_w,target_w,Cxy,2)
        similarity = []
        for i in range(Xembed.shape[0]):
            similarity.append(cos_sim(Xembed[i],Yembed))
        best_source = np.argmax(similarity)
        best_source_w = source_w[best_source]
        best_source_w = best_source_w.reshape(3,400)
        action = np.argmax([Q(state,a,best_source_w) for a in range(nA)])
    return action

# =============================================================================
# def policy(state, weight, epsilon=0.1):
# 	A = np.ones(nA,dtype=float) * epsilon/nA
# 	best_action =  np.argmax([Q(state,a,w) for a in range(nA)])
# 	A[best_action] += (1.0-epsilon)
# 	sample = np.random.choice(nA,p=A)
# 	return sample
# =============================================================================

# OPTIONAL check gradients 
def check_gradients(index,state,next_state,next_action,weight,reward):
	ew1 = np.array(weight, copy=True) 
	ew2 = np.array(weight, copy=True)  
	epsilon = 1e-6
	ew1[action][index] += epsilon
	ew2[action][index] -= epsilon
	test_target_1 = reward + discount_factor * Q(next_state,next_action,ew1)		
	td_error_1 = target - Q(state,action,ew1)
	test_target_2 = reward + discount_factor * Q(next_state,next_action,ew2)		
	td_error_2 = target - Q(state,action,ew2)
	grad = (td_error_1 - td_error_2) / (2 * epsilon)
	return grad[0]

# Our main training loop
total_reward_value = 0
average_reward = []

for episode in range(num_episodes):
    state = env.reset()
    state = featurize_state(state)
    
    while True:
        env.render()
		# Sample from our policy
        if (episode+1)>=5:
            action = entropy_policy(state,w)
        else:
            action = policy(state,w)
        #action = ma_policy(state,w)
        #action = policy(state,w)
        # Staistic for graphing
        plt_actions[action] += 1
		# Step environment and get next state and make it a feature
        next_state, reward, done, _ = env.step(action)
        next_state = featurize_state(next_state)
		# Figure out what our policy tells us to do for the next state
        next_action = policy(next_state,w)
		# Statistic for graphing
        total_reward_value = total_reward_value+reward
		# Figure out target and td error
        target = reward + discount_factor * Q(next_state,next_action,w)
        td_error = Q(state,action,w) - target
		# Find gradient with code to check it commented below (check passes)
        dw = (td_error).dot(state)
		#for i in range(4):
		#	print("First few gradients")
		#	print(str(i) + ": " + str(check_gradients(i,state,next_state,next_action,w,reward)) + " " + str(dw[i]))
		# Update weight
        w[action] -= alpha * dw
        if done:
            break
		# update our state
        state = next_state
    print('{} episode over'.format(episode+1))
    print('average reward {}'.format(total_reward_value/(episode+1)))
    average_reward.append(total_reward_value/(episode+1))
print(average_reward)
# =============================================================================
# last_w = []
# for i in range(w.shape[0]):
#     mid = []
#     for j in range(w.shape[1]):
#         mid.append(w[i][j])
#     last_w.append(mid)
# last_w = np.array(last_w)
# np.save('2.npy',last_w)
# =============================================================================

env.close()

# =============================================================================
# def plot_cost_to_go_mountain_car(num_tiles=20):
#     x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
#     y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
#     X, Y = np.meshgrid(x, y)
#     Z = np.apply_along_axis(lambda _: -np.max([Q(featurize_state(_),a,w) for a in range(nA)]), 2, np.dstack([X, Y]))
#     fig = plt.figure(figsize=(10, 5))
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                            cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
#     ax.set_xlabel('Position')
#     ax.set_ylabel('Velocity')
#     ax.set_zlabel('Value')
#     ax.set_title("Mountain \"Cost To Go\" Function")
#     fig.colorbar(surf)
#     plt.show()
# 
# # Show bar graph of actions chosen
# plt.bar(np.arange(nA),plt_actions)
# plt.figure()
# # Plot the reward over all episodes
# plt.plot(np.arange(num_episodes),episode_rewards)
# plt.show()
# # plot our final Q function
# plot_cost_to_go_mountain_car()
# env.close()
# =============================================================================
