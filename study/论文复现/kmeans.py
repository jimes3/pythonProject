import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KMeans:
    def __init__(self, n_clusters, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def initialize_centroids(self, X):
        random_indices = np.random.choice(len(X), size=self.n_clusters, replace=False)
        centroids = X[random_indices]
        return centroids

    def assign_clusters(self, X, centroids):
        clusters = np.zeros(len(X))
        for i, x in enumerate(X):
            distances = [euclidean_distance(x, centroid) for centroid in centroids]
            cluster = np.argmin(distances)
            clusters[i] = cluster
        return clusters

    def update_centroids(self, X, clusters):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for cluster in range(self.n_clusters):
            cluster_points = X[clusters == cluster]
            centroid = np.mean(cluster_points, axis=0)
            centroids[cluster] = centroid
        return centroids

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iters):
            old_centroids = np.copy(self.centroids)
            clusters = self.assign_clusters(X, self.centroids)
            self.centroids = self.update_centroids(X, clusters)
            if np.all(old_centroids == self.centroids):
                break

    def predict(self, X):
        clusters = self.assign_clusters(X, self.centroids)
        return clusters
# 创建数据集
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# 创建KMeans对象并进行聚类
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# 获取聚类结果
labels = kmeans.predict(X)
centroids = kmeans.centroids

# 输出聚类结果和质心坐标
print("聚类标签：", labels)
print("质心坐标：", centroids)
