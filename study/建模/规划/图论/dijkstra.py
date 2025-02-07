import numpy as np
import math

def dijkstra(distance_matrix, start):
    num_nodes = len(distance_matrix)
    # 初始化距离和前驱字典
    distances = {node: math.inf for node in range(num_nodes)}
    predecessors = {node: None for node in range(num_nodes)}
    # 将起始节点距离设为0
    distances[start] = 0
    # 追踪已经确定最短路径的节点
    visited = set()
    while len(visited) < num_nodes:
        # 找到当前未访问节点中距离最小的节点
        current_distance = math.inf
        current_node = None
        for node in range(num_nodes):
            if distances[node] < current_distance and node not in visited:
                current_distance = distances[node]
                current_node = node
        if current_node is None:
            break
        # 将当前节点标记为已访问
        visited.add(current_node)
        # 更新当前节点的相邻节点的距离
        for neighbor in range(num_nodes):
            if distance_matrix[current_node, neighbor] > 0:
                distance = current_distance + distance_matrix[current_node, neighbor]
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
    return distances, predecessors

# 示例距离矩阵
distance_matrix = np.array([
    [0, 5, 2, 0, 0],
    [5, 0, 0, 1, 6],
    [2, 0, 0, 3, 0],
    [0, 1, 3, 0, 4],
    [0, 6, 0, 4, 0]
])

start_node = 0  # 假设从节点0开始

# 使用Dijkstra算法查找最短路径
distances, predecessors = dijkstra(distance_matrix, start_node)

print("最短距离:")
for node, distance in distances.items():
    print(f"从节点 {start_node} 到节点 {node} 的最短距离为: {distance}")

print("\n前驱节点:")
for node, predecessor in predecessors.items():
    print(f"节点 {node} 的前驱节点是: {predecessor}")
