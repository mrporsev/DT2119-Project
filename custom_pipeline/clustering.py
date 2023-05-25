import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def adjacency_matrix(data, measure):
    # data is an N x D ndarray
    # measure is a similarity measure function

    n = data.shape[0]

    A = np.empty((n, n))

    for i in range(n):
        for j in range(n):
            A[i, j] = measure(data[i, :].flatten(), data[j, :].flatten())

    return A


def cosine_similarity(a, b):
    # a and b are equal length ndarrays
    
    return (a.reshape(1, -1) @ b.reshape(-1, 1)) / np.sqrt(np.sum(np.square(a)) * np.sum(np.square(b)))

def something(a, b):
    return np.maximum(0, (cosine_similarity(a, b)+1)/2)

def euclid_dist(a, b): # this is not euqlidian distance, very bad name
    #return 1/np.sqrt(1+np.sum(np.square(a-b)))
    #return 1/(1+np.sum(np.square(a-b)))
    return 1/np.power(1+np.sum(np.square(a-b)), 2)

def normalized_laplacian_matrix(A):
    # A is an adjacency matrix

    D = np.sum(A, axis=1)
    inv_sqrt_D = np.power(D, -0.5)
    L = inv_sqrt_D.reshape(-1, 1) * A * inv_sqrt_D.reshape(1, -1)

    return L



data = []
data_colors = []
data.append(np.random.multivariate_normal([10, 0], [[1, 0],[0, 1]], 100))
data_colors.append(0*np.ones(100))
data.append(np.random.multivariate_normal([3, 0.5], [[0.5, 0],[0, 0.5]], 100))
data_colors.append(1*np.ones(100))
data.append(np.random.multivariate_normal([1, -1.5], [[0.5, 0],[0, 0.01]], 100))
data_colors.append(2*np.ones(100))
data.append(np.random.multivariate_normal([1, -1.5], [[0.01, 0],[0, 0.5]], 100))
data_colors.append(3*np.ones(100))
data.append(np.random.multivariate_normal([8, -1.5], [[0.5, 0],[0, 1]], 100))
data_colors.append(4*np.ones(100))

data = np.concatenate(data)
data_colors = np.concatenate(data_colors)
p = np.random.permutation(data.shape[0])
data = data[p, :]
data_colors = data_colors[p]

A = adjacency_matrix(data, euclid_dist)

plt.imshow(A)
plt.show()

L = normalized_laplacian_matrix(A)

plt.imshow(L)
plt.show()

w, v = np.linalg.eig(L)

eigenvalues = np.real(w)
eigenvalues = eigenvalues[eigenvalues>0.05]
#eigenvectors = np.real(v[:, 0:len(eigenvalues)])
#print(np.diff(eigenvalues))
num_vectors = np.argmin(np.diff(eigenvalues)) + 1
eigenvectors = np.real(v[:, 0:num_vectors])
plt.scatter(range(len(eigenvalues)), eigenvalues)
plt.show()

#print(eigenvalues)
#print(eigenvectors)

norm_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=1, keepdims=True)

plt.scatter(norm_eigenvectors[:, 0], norm_eigenvectors[:, 1])
plt.show()
plt.scatter(norm_eigenvectors[:, 1], norm_eigenvectors[:, 2])
plt.show()

predictions = KMeans(n_clusters=norm_eigenvectors.shape[1], n_init=5).fit_predict(norm_eigenvectors)

plt.figure(1)
plt.scatter(data[:, 0], data[:, 1], c=predictions)
plt.title("prediction")

plt.figure(2)
plt.scatter(data[:, 0], data[:, 1], c=data_colors)
plt.title("ground truth")
plt.show()
