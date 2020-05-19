import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

A = np.array([
    [3.1, 2.3],
    [2.3, 4.2],
    [3.9, 3.5],
    [3.7, 6.4],
    [4.8, 1.9],
    [8.3, 3.1],
    [5.2, 7.5],
    [4.8, 4.7],
    [3.5, 5.1],
    [4.4, 2.9],
])

k = 3

test_data = [3.3, 2.9]

plt.figure()
plt.title('Input data')
plt.scatter(A[:, 0], A[:, 1], marker='o', s=100, color='black')

knn_model = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(A)
distances, indices = knn_model.kneighbors([test_data])

print("\nK Nearest Neighbors:")
for rank, index in enumerate(indices[0][:k], start=1):
    print(str(rank) + " is", A[index])

plt.figure()
plt.title('Nearest neighbors')
plt.scatter(A[:, 0], A[:, 1], marker='o', s=100, color='k')
plt.scatter(A[indices][0][:][:, 0],
            A[indices][0][:][:, 1],
            marker='o',
            s=250,
            color='k',
            facecolors='none')
plt.scatter(test_data[0], test_data[1], marker='x', s=100, color='k')
plt.show()
