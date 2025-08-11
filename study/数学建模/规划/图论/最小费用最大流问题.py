from collections import deque
def spfa(graph, source,sink):       # SPFA 最短路径算法
    n = len(graph)
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
            if graph[u][v] != 0:  # 0 表示不可达，只处理非零值作为边权重
                # 如果通过节点 u 可以获得更短的路径，则进行更新
                if dist[u] + graph[u][v] < dist[v]:
                    dist[v] = dist[u] + graph[u][v]  # 更新节点 v 的最短距离
                    parent[v] = u              # 设置节点 v 的父节点为 u
                    # 如果节点 v 不在队列中，则将其放入队列，并标记为在队列中
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True
        if u == sink: # 如果找到终点
            break   #结束算法
    return dist, parent
def get_path(sink, parent):
    path = []
    while sink != -1:
        path.append(sink)
        sink = parent[sink]
    return path[::-1]


if __name__ == "__main__":
    cost_graph = [
        [0, 5, 2, 0, 0],
        [0, 0, 0, 0, 6],
        [0, 4, 0, 3, 0],
        [0, 0, 0, 0, 4],
        [0, 0, 0, 0, 0]
    ]
    capacity_graph = [
        [0, 9, 7, 0, 0],
        [0, 0, 0, 0, 6],
        [0, 7, 0, 6, 0],
        [0, 0, 0, 0, 9],
        [0, 0, 0, 0, 0]
    ]
    source = 0
    sink = 4
    s = 0
    print('上面的优先级高')
    while True:
        dist, parent = spfa(cost_graph,source,sink)
        path = get_path(sink,parent)
        if len(path)==1: #找不到路径就跳出循环
            print('最大流：',s)
            break
        a = min([capacity_graph[path[i]][path[i+1]] for i in range(len(path)-1)])
        s += a
        for i in range(len(path)-1):  #更新容量矩阵
            capacity_graph[path[i]][path[i+1]] -= a
        # 如果费用与量相关，将下面的值乘以a即可，矩阵为1单位的费用 #
        for i in range(len(path)-1):  #更新费用矩阵，添加反向边
            cost_graph[path[i+1]][path[i]] = -cost_graph[path[i]][path[i+1]]
            if a == path[i]:     #更新费用矩阵，删掉无容量边
                cost_graph[path[i]][path[i+1]] = 0
        print('流量:',a,'路径:',path)