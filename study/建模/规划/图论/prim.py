
def prim_mst(graph, num_nodes):
    # 初始化一个空的最小生成树
    mst = []
    # 选择一个起始节点，这里假设起始节点为0
    start_node = 0
    # 将起始节点添加到最小生成树中
    mst.append(start_node)
    # 初始化一个集合来保存已经加入最小生成树的节点
    visited = set([start_node])
    # 初始化一个列表来保存最小生成树的边信息
    edges = []
    # 重复以下步骤，直到最小生成树包含所有节点
    while len(mst) < num_nodes:
        # 初始化当前轮次的最小边的权值、起始节点和目标节点
        min_weight = float('inf')
        next_node = None
        target_node = None
        # 遍历已加入最小生成树的节点
        for node in mst:
            # 在图中找到与当前节点相连的所有边
            for neighbor, weight in enumerate(graph[node]):
                # 如果相邻节点不在已加入最小生成树的节点集合中，且权值更小
                if neighbor not in visited and weight > 0 and weight < min_weight:
                    # 更新最小边的权值、起始节点和目标节点
                    min_weight = weight
                    next_node = node
                    target_node = neighbor
        # 将找到的最小边加入最小生成树的边信息列表
        edges.append((next_node, target_node, min_weight))
        # 将目标节点加入最小生成树的节点集合
        mst.append(target_node)
        visited.add(target_node)
    return edges

# 示例用法
if __name__ == "__main__":
    # 定义图的邻接矩阵，表示节点之间的边及其权值
    # 这里假设节点之间的距离用正整数表示，无连接的节点用0表示
    graph = [
        [0, 4, 3, 0, 0, 0],
        [4, 0, 2, 1, 0, 0],
        [3, 2, 0, 4, 0, 0],
        [0, 1, 4, 0, 2, 0],
        [0, 0, 0, 2, 0, 6],
        [0, 0, 0, 0, 6, 0],
    ]
    result = prim_mst(graph, len(graph))
    # 输出最小生成树的边信息
    print("最小生成树的边信息:")
    for edge in result:
        print(f"节点{edge[0]}和节点{edge[1]}之间的边权值为{edge[2]}")
