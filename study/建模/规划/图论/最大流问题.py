from collections import defaultdict
#ford_fulkerson算法
class Graph:
    def __init__(self):
        self.graph = defaultdict(dict)
    def add_edge(self, u, v, capacity):
        # 添加正向边
        self.graph[u][v] = capacity
        # 添加反向边
        self.graph[v][u] = 0  # 反向边的初始流量为0
    def ford_fulkerson(self, source, sink):
        def bfs(source, sink, parent):
            # 使用BFS寻找增广路径
            visited = set()
            queue = [(source, float('inf'))]
            while queue:
                u, flow = queue.pop(0)
                visited.add(u)
                for v, capacity in self.graph[u].items():
                    if v not in visited and capacity > 0:
                        parent[v] = (u, capacity)
                        min_flow = min(flow, capacity)
                        if v == sink:
                            return min_flow
                        queue.append((v, min_flow))
            return 0
        max_flow = 0
        parent = {}
        my_dict = {} #字典，存储路径与流量
        while True:
            min_flow = bfs(source, sink, parent)
            if min_flow == 0:
                break
            v = sink
            path = [v]
            while v != source:
                u, capacity = parent[v]
                # 更新正向边和反向边上的流量
                self.graph[u][v] -= min_flow
                self.graph[v][u] += min_flow
                v = u
                path.insert(0, v)
            max_flow += min_flow
            if str(path) in my_dict:
                my_dict[str(path)] += min_flow
            else:
                my_dict[str(path)] = min_flow
        return max_flow, my_dict

_graph = {
    'A': {'B': 6, 'C': 8},
    'B': {'D': 8,'E': 6},
    'C': {'D': 8,'B': 6},
    'D': {'F': 8},
    'E': {'F': 6,'D': 8},
    'F': {}  #终点不必写
}
# 创建图对象并添加边
graph = Graph()
for u, edges in _graph.items():
    for v, capacity in edges.items():
        graph.add_edge(u, v, capacity)
max_flow, my_dict = graph.ford_fulkerson(source='A', sink='F')
print("最大流量为:", max_flow)
print("所有最大流路径和对应的流量为:")
for key, value in my_dict.items():
    print('路径：',key,'流量：', value)
