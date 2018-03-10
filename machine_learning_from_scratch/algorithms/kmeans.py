import numpy as np


def kmeans(X, k, num_iters=10):
    """
    Uses kmeans to compute k centroids and returns a cluster for each data point.

    Parameters
    ----------
    X : np.array
        Data in the shape (examples, features).
    k : int
        Number of clusters.
    num_iters : int
        Number of iterations.

    Returns
    -------
    clusters : np.array
        The cluster for each data point.
    centroids: np.array
        The centroids of each cluster.

    """

    # Initialise centroids
    num_feats = X.shape[1]
    centroids = np.random.randn(k, num_feats)

    for _ in range(num_iters):
        # Compute distance from points to each centroid
        D = compute_distances(X, centroids)

        # Allocate a point to each cluster
        clusters = D.argmin(axis=1)

        # Compute new centroids
        for i in range(k):
            centroids[i, :] = X[clusters == i, :].mean(axis=0)

    return clusters, centroids


def euclidean_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


def compute_distances(X, centroids):
    """
    Compute euclidean distance between each data point and each centroid.
    """
    num_examples = X.shape[0]
    k = centroids.shape[0]
    # Compute distance from points to each centroid
    D = np.zeros(shape=(num_examples, k))
    for i in range(k):
        D[:, i] = euclidean_dist(X, centroids[i, :])
    return D


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs

    X, y = make_blobs(centers=2)
    clusters, centroids = kmeans(X, k=2, num_iters=10)

    # Plot
    plt.scatter(X[:, 0], X[:, 1], c=clusters, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=100, c=np.arange(0, centroids.shape[0]))
    plt.show()
