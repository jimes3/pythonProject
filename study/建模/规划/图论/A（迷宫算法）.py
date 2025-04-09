import numpy as np
import math

def heuristic(node, goal, pos):
    node_x, node_y = pos[node+1]
    goal_x, goal_y = pos[goal+1]
    return abs(node_x - goal_x) + abs(node_y - goal_y)

def astar(distance_matrix, start, goal,pos):
    num_nodes = len(distance_matrix)
    # 初始化距离和前驱字典
    distances = {node: math.inf for node in range(num_nodes)}
    predecessors = {node: None for node in range(num_nodes)}
    # 将起始节点距离设为0
    distances[start] = 0
    # 追踪已经确定最短路径的节点
    visited = set()
    while len(visited) < num_nodes:
        # 找到当前未访问节点中 f 值最小的节点
        current_f = math.inf
        current_node = None
        for node in range(num_nodes):
            f = distances[node] + heuristic(node, goal, pos)
            if f < current_f and node not in visited:
                current_f = f
                current_node = node
        if current_node is None:
            break
        # 将当前节点标记为已访问
        visited.add(current_node)
        # 更新当前节点的相邻节点的距离
        for neighbor in range(num_nodes):
            if distance_matrix[current_node, neighbor] > 0:
                distance = distances[current_node] + distance_matrix[current_node, neighbor]
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
    return distances, predecessors

# 距离矩阵，没有直接相连则为0
distance_matrix = np.array([[0, 6, 1, 5, 0, 0],
                            [6, 0, 5, 0, 3, 0],
                            [1, 5, 0, 5, 6, 4],
                            [5, 0, 5, 0, 0, 2],
                            [0, 3, 6, 0, 0, 6],
                            [0, 0, 4, 2, 6, 0]])
# 每个点的坐标，算法会考虑与终点的距离
pos={1:(1,8),2:(4,10),3:(11,11),4:(14,8),5:(5,7),6:(10,6),7:(3,5),8:(6,4),9:(8,4),10:(14,5),11:(2,3),12:(5,1),13:(8,1),14:(13,3)}

start_node = 0  # 假设从节点0开始
goal_node = 4   # 假设目标节点是节点4

# 使用A*算法查找最短路径
distances, predecessors = astar(distance_matrix, start_node, goal_node,pos)

print("最短距离:")
for node, distance in distances.items():
    print(f"从节点 {start_node} 到节点 {node} 的最短距离为: {distance}")

print("\n前驱节点:")
for node, predecessor in predecessors.items():
    print(f"节点 {node} 的前驱节点是: {predecessor}")
# 回溯并输出最短路径
path = [goal_node]
current_node = goal_node
while current_node != start_node:
    current_node = predecessors[current_node]
    path.append(current_node)
path.reverse()
print("最短路径:", "->".join(str(node) for node in path))