from collections import deque

def spfa_shortest_path(distance_matrix, source):
    n = len(distance_matrix)
    dist = [float('inf')] * n  # 节点到源点的最短距离
    parent = [-1] * n          # 记录路径中的父节点
    in_queue = [False] * n     # 判断节点是否在队列中
    dist[source] = 0           # 初始节点到自身的距离为0
    queue = deque([source])    # 创建一个双向队列，并将初始节点放入队列
    in_queue[source] = True    # 标记初始节点已在队列中
    while queue:  # 开始 SPFA 算法
        u = queue.popleft()    # 从队列头部取出节点 u
        in_queue[u] = False    # 将节点 u 标记为不在队列中
        # 遍历节点 u 的所有邻接节点
        for v in range(n):
            if distance_matrix[u][v] > 0:  # 0 表示不可达，只处理非零值作为边权重
                # 如果通过节点 u 可以获得更短的路径，则进行更新
                if dist[u] + distance_matrix[u][v] < dist[v]:
                    dist[v] = dist[u] + distance_matrix[u][v]  # 更新节点 v 的最短距离
                    parent[v] = u                               # 设置节点 v 的父节点为 u
                    # 如果节点 v 不在队列中，则将其放入队列，并标记为在队列中
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True
    return dist, parent
def get_path(node, parent):# 输出最短路径
    path = []
    while node != -1:
        path.append(node)
        node = parent[node]
    return ' -> '.join(map(str, path[::-1]))

# 示例
if __name__ == "__main__":
    # 示例距离矩阵，0 表示不可达
    distance_matrix = [
        [0, 5, 2, 0, 0],
        [5, 0, 0, 1, 6],
        [2, 0, 0, 3, 0],
        [0, 1, 3, 0, 4],
        [0, 6, 0, 4, 0]
    ]
    source = 0  # 源点
    dist, parent = spfa_shortest_path(distance_matrix, source)
    for node in range(len(distance_matrix)):
        if dist[node] < float('inf'):
            print(f"从节点 {source} 到节点 {node} 的最短路径为：{get_path(node, parent)}，最短距离为：{dist[node]}")
        else:
            print(f"从节点 {source} 到节点 {node} 不可达")
