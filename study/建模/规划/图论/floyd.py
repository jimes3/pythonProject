import numpy as np
def floyd(graph):
    n = len(graph)  #得到矩阵大小指标
    dist = np.zeros((n,n)) #初始化
    for i in range(n):  #初始化
        for j in range(n):  #将矩阵部分0换为INF
            if i != j and graph[i][j]==0:
                dist[i][j] = float('inf')
            else:
                dist[i][j] = graph[i][j]
    a = [[None for _ in range(n)] for _ in range(n)]
    for k in range(n):  #算法核心
        for i in range(n):
            for j in range(n):  #保存路径
                l = dist[i][k] + dist[k][j]
                if dist[i][j] > l:
                    dist[i][j] = l
                    a[i][j]=k
    def r_path(path, start, end): #回溯路径
        intermediate = path[start][end]
        if intermediate == None:  # 如果两节点之间没有直接路径，返回空
            return [start, end]
        else:  # 否则，递归构建路径
            path_1 = r_path(path, start, intermediate)
            path_2 = r_path(path, intermediate, end)
        return path_1 + path_2[1:]  # 去掉重复的中间节点
    b = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            b[i][j] = r_path(a,i,j)
            print(f'{i}到{j}的最短路径为{b[i][j]},路程为{dist[i][j]}')
    return dist

graph = [[0, 6, 1, 5, 0, 0],
         [6, 0, 5, 0, 3, 0],
         [1, 5, 0, 5, 6, 4],
         [5, 0, 5, 0, 0, 2],
         [0, 3, 6, 0, 0, 6],
         [0, 0, 4, 2, 6, 0]]
result = floyd(graph)
print('最短路程:\n',result)