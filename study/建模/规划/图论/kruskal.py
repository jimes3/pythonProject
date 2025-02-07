class UnionFind:  #并查操作
    def __init__(self, n):
        self.parent = [i for i in range(n)]  # 初始化每个节点的父节点为自己
        self.rank = [0] * n  # 初始化每个节点的秩（用于优化Union操作）
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 使用路径压缩来找到节点的根父节点
        return self.parent[x]
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False  # 如果两个节点的根父节点相同，说明它们已经在同一个集合中，不需要合并
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y  # 将rank较小的节点作为rank较大的节点的子节点
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x  # 两个节点rank相同，任意选一个作为子节点，并将其rank增加1
            self.rank[root_x] += 1
        return True  # 合并成功，返回True
def kruskal(distance_matrix):
    num_nodes = len(distance_matrix)
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if distance_matrix[i][j] != 0:  # 只添加有连接的边
                edges.append((i, j, distance_matrix[i][j]))  # 用元组表示边的信息：(节点1, 节点2, 权重)
    edges.sort(key=lambda x: x[2])  # 按权重升序排序边
    mst_edges = []  # 用于存储最小生成树的边
    uf = UnionFind(num_nodes)  # 初始化并查集
    for edge in edges:
        src, dest, weight = edge
        if uf.union(src, dest):
            mst_edges.append((src, dest))  # 如果成功将边的两个节点合并到同一个集合中，将该边添加到最小生成树中
    return mst_edges  # 返回最小生成树的边列表

# 示例用法：
if __name__ == "__main__":
    distance_matrix = [
        [0, 4, 3, 0, 0, 0],
        [4, 0, 2, 1, 0, 0],
        [3, 2, 0, 4, 0, 0],
        [0, 1, 4, 0, 2, 0],
        [0, 0, 0, 2, 0, 6],
        [0, 0, 0, 0, 6, 0],
    ]

    mst_edges = kruskal(distance_matrix)
    total_weight = sum(distance_matrix[src][dest] for src, dest in mst_edges)
    print("最小生成树的边:", mst_edges)
    print("最小生成树的总权重:", total_weight)
