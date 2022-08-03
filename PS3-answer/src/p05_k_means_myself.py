from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import random
import pprint as pp

def compress(A,K,max_iter,image_name):
    """
    A: image matrix
    K: number of means
    max_iter: max iteration times
    """
    # initialize mu
    mu = np.zeros((K,3),dtype=int)
    for i in range(16):
        mu[i] = np.array(A[random.randint(0, A.shape[0]-1),random.randint(0, A.shape[1]-1)])
    
    # initilize centroids
    centroids = np.zeros((A.shape[0],A.shape[1]),dtype=int)
    # update centroids and mu
    for iter in range(max_iter):
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                centroids[i,j] = np.argmin(np.linalg.norm(A[i,j]-mu,axis=1))
        for k in range(K):
            if np.sum(centroids==k):
                mu[k] = np.mean(A[centroids==k],axis=0)
    # update image
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i,j] = mu[centroids[i,j]]
    plt.imshow(A)
    plt.savefig(fname='../output/{}.png'.format(image_name), transparent=True, format='png', bbox_inches='tight')

if __name__ == "__main__":
    max_iter = 30
    K = 16

    A = imread('../data/peppers-small.tiff')
    plt.imshow(A)
    plt.show()
    compress(A, K, max_iter, 'peppers-small')

    A = imread('../data/peppers-large.tiff')
    plt.imshow(A)
    plt.show()
    compress(A, K, max_iter, 'peppers-large')

# K=16

# A = imread('../data/peppers-small.tiff')
# plt.imshow(A)
# plt.show()

# mu = np.zeros((K,3),dtype=int)
# # print(mu)
# for i in range(16):
#     mu[i] = np.array(A[random.randint(0, A.shape[0]-1),random.randint(0, A.shape[1]-1)])
# # mu = np.array(mu)
# # pp.pprint(mu)
# # pp.pprint(A[0][0])
# # pp.pprint(A[0][0]-mu)
# centroids = np.zeros((A.shape[0],A.shape[1]),dtype=int)
# for iter in range(30):
#     for i in range(A.shape[0]):
#         for j in range(A.shape[1]):
#             centroids[i,j] = np.argmin(np.linalg.norm(A[i,j]-mu,axis=1))
#     pp.pprint(centroids)
#     for k in range(K):
#         if np.sum(centroids==k):
#             mu[k] = np.mean(A[centroids==k],axis=0)
# for i in range(A.shape[0]):
#     for j in range(A.shape[1]):
#         A[i,j] = mu[centroids[i,j]]
#         print(A[i,j])
# plt.imshow(A)
# # plt.show()
# plt.savefig('../output/peppers-small-compressed.tiff')