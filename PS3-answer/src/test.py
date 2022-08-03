from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import random
import pprint as pp

K=16 

A = imread('../data/peppers-small.tiff')
plt.imshow(A)
plt.show()

mu = np.zeros((16,3),dtype=int)
# print(mu)
for i in range(16):
    mu[i] = np.array(A[random.randint(0, A.shape[0]-1),random.randint(0, A.shape[1]-1)])

# A = imread('../data/peppers-large.tiff')
# c=np.random.randint(0,A.shape[0],size=2)
# print(np.linalg.norm(A[0,0]-mu,axis=1))
# print(np.argsort(np.linalg.norm(A[0,0]-mu,axis=1)))
c_i = np.zeros((A.shape[0],A.shape[1]),dtype=int)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        c_i[i,j] = np.argsort(np.linalg.norm(A[i,j]-mu,axis=1))[0]
# print(c_i==3)
for k in range(K):
    if np.sum(c_i==k):
        print(A[c_i==k].shape)
        mu[k] = np.mean(A[c_i==k],axis=0)
        print(mu[k])