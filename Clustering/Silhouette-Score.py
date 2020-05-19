import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics

X, y_true = make_blobs(n_samples=500,
                       centers=4,
                       cluster_std=0.40,
                       random_state=0)

scores = []
values = np.arange(2, 10)

for num_clusters in values:
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)
    score = metrics.silhouette_score(X,
                                     kmeans.labels_,
                                     metric='euclidean',
                                     sample_size=len(X))

    print("\nNumber of clusters =", num_clusters)
    print("Silhouette score =", score)
    scores.append(score)

num_clusters = np.argmax(scores) + values[0]
print('\nOptimal number of clusters =', num_clusters)
