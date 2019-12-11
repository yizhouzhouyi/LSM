# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:20:58 2018
@author: yizhou
"""

from scipy.linalg import block_diag, eigh, svd
from scipy.sparse.csgraph import laplacian
import numpy as np
import pandas as pd
# =============================================================================
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# =============================================================================

'''transform dataframe q into (1,100) vector'''
def transformq(state_dict,source_task):
    source_q = np.zeros((100,4))
    for i in range(source_task.shape[0]):
        source_q[state_dict[source_task.index[i]],:] = source_task.loc[source_task.index[i],:]
    source_q = source_q.reshape((1,-1))
    return source_q

'''compute the cosine similarity between two vectors'''
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
    nx, dx = X.shape  #X
    ny, dy = Y.shape  #
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


# =============================================================================
# Cxy = compute_cxy(source_q,task_q)
# Xembed,Yembed = low_rank_align(source_q,task_q,Cxy,d=2,mu=0.8)
# print(Xembed)
# print(Yembed)
# 
# simi = []
# for j in range(Xembed.shape[0]):
#     simi.append(cos_sim(Xembed[j],Yembed))
# print(simi)
# =============================================================================
# =============================================================================
# X = [[1.2,0.8,6.2,1.5],
#      [2.05,7.01,0.80,2.0],
#      [0.1,-1.22,4.2,2.00]]
# X = np.array(X)
# Y = [[2.9,3.0],
#      [-0.2,3.5]]
# Y = np.array(Y)
# Cxy = np.eye(X.shape[0],Y.shape[0])
# print(Cxy)
# Xembed,Yembed = low_rank_align(X,Y,Cxy,d=2,mu=0.8)
# print(Xembed)
# print(Yembed)
# =============================================================================
