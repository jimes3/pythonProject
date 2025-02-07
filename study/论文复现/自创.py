import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix


node = np.array([(456, 320),(228, 0),(912, 0),(0, 80), (114, 80), (570, 160),(798, 160),(342, 240),(684, 240),
                 (570, 400),(912, 400), (114, 480),(228, 480),(342, 560),(684, 560),(0, 640),(798, 640),
                  (228, 93), (236, 93), (236, 101), (228, 101), (228, 109),(228, 117), (228, 125), (220, 125)])
dist_node = distance_matrix(node, node)

#获得投影点
def shadow(node, a, b):
    ab_square = a**2 + b**2
    aa = a**2
    bb = b**2
    ab = a * b
    transformation_matrix0 = np.array([[bb, -ab], [-ab, aa]]) / ab_square
    transformation_matrix1 = np.array([[aa, ab], [ab, bb]]) / ab_square
    shadow0 = np.dot(node, transformation_matrix0)
    shadow1 = np.dot(node, transformation_matrix1)
    return shadow0, shadow1

def is_obtuse_angle(point1, point2, point3):
    vector1 = np.array(point2) - np.array(point1)
    vector2 = np.array(point3) - np.array(point2)

    dot_product = np.dot(vector1, vector2)

    if dot_product < 0:
        return True
    else:
        return False

def is_duandian(dian,node,dist_node):
    split_ = []
    for h in range(len(dian)-1):
        index0 = np.where(np.all(node == dian[h], axis=1))[0]
        index1 = np.where(np.all(node == dian[h+1], axis=1))[0]
        if dist_node[index0,index1] == 0:
            split_.append(h+1)
    return split_

shadow0, shadow1 = shadow(node,1,1)

sd0 = DBSCAN(eps=30, min_samples=3)
sd0.fit(shadow0)
pred0 = sd0.labels_

sd1 = DBSCAN(eps=30, min_samples=5)
sd1.fit(shadow1)
pred1 = sd1.labels_

plt.scatter(shadow0[:, 0],shadow0[:, 1], c=pred0)
plt.scatter(shadow1[:, 0],shadow1[:, 1], c=pred1)

plt.title('Sklearn DBSCAN')
plt.show()

# 获取每个团的数据
clusters = []
unique_labels = np.unique(pred0)
for label in unique_labels:
    if label != -1:  # 排除噪声点
        cluster = node[pred0 == label]
        clusters.append(cluster)

for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}:")
    print(cluster)
    for v in range(len(cluster)-2):
        is_obtuse = is_obtuse_angle(cluster[v], cluster[v+1], cluster[v+2])
    split = is_duandian(cluster,node,dist_node)

def split_cluster(split,cluster):
    arrays = []
    start = 0
    for index in split:
        array = cluster[start:index]
        arrays.append(array)
        start = index
    # 处理最后一个分割点到数组末尾的部分
    array = cluster[start:]
    arrays.append(array)
