import numpy as np
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, eps, min_samples, max_cluster_weight):
        self.eps = eps
        self.min_samples = min_samples
        self.max_cluster_weight = max_cluster_weight

    def fit(self, X, weights):
        self.labels_ = np.full(shape=X.shape[0], fill_value=-1, dtype=int)  # -1 表示噪声点
        self.cluster_id_ = 0

        for i in range(X.shape[0]):
            if self.labels_[i] != -1:
                continue
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1  # 噪声点
            else:
                self._expand_cluster(X, i, neighbors, weights[i])

    def _region_query(self, X, point_index):
        """找到eps邻域内的点"""
        points = X - X[point_index]
        distances = np.sqrt(np.sum(points**2, axis=1))
        neighbors = np.where(distances < self.eps)[0]
        return neighbors

    def _expand_cluster(self, X, point_index, neighbors, point_weight):
        """扩展簇"""
        if point_weight > self.max_cluster_weight:
            self.labels_[point_index] = -1  # 超过权值和限制，标记为噪声点
            return

        self.cluster_id_ += 1
        self.labels_[point_index] = self.cluster_id_
        cluster_weight = point_weight
        seeds = [point_index]

        while len(seeds) > 0:
            current_point = seeds.pop(0)
            current_weight = weights[current_point]
            if cluster_weight + current_weight > self.max_cluster_weight:
                continue  # 超过权值和限制，不加入簇

            neighbors = self._region_query(X, current_point)
            for neighbor in neighbors:
                if self.labels_[neighbor] != -1:
                    continue
                neighbor_weight = weights[neighbor]
                if cluster_weight + neighbor_weight > self.max_cluster_weight:
                    continue  # 超过权值和限制，不加入簇

                self.labels_[neighbor] = self.cluster_id_
                cluster_weight += neighbor_weight
                seeds.append(neighbor)

    def plot_clusters(self, X):
        plt.scatter(X[:, 1], X[:, 0], c=self.labels_, cmap='viridis', marker='o')
        plt.scatter(107.078539,33.052466)
        plt.title('Weighted DBSCAN Clustering')
        plt.show()

# 生成示例数据
X=np.array([[33.076266, 107.060763], [32.853819, 107.11977], [32.855425, 107.02023], [32.476447, 107.24196], [33.413498, 107.192686], [33.335921, 106.155235], [33.335242, 105.966226], [33.206888, 106.42617], [33.32215, 105.822041], [33.222552, 106.083149], [33.51653, 105.911839], [33.266853, 106.254729], [33.473713, 106.043551], [32.43499, 107.553329], [32.465878, 107.483882], [32.714549, 107.885112], [32.526814, 107.570597], [32.595013, 107.717875], [32.731158, 107.991084], [32.642829, 107.753562], [33.544629, 107.82695], [33.54876, 107.993614], [33.45927, 107.97293], [33.47337, 108.11846]
            ])
weights = [8.28, 1.12, 0.49, 0.36, 0.52, 2.75, 0.4, 0.72, 0.97, 0.69, 0.27, 1.01, 0.21, 0.97, 0.43, 0.3, 0.36, 1.21, 0.66, 0.45, 0.23, 0.24, 0.26, 0.23]
# 创建DBSCAN对象
dbscan = DBSCAN(eps=0.3, min_samples=5,max_cluster_weight=10)

# 进行聚类
dbscan.fit(X,weights)

# 可视化结果
dbscan.plot_clusters(X)