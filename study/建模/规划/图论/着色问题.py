def graph_coloring(graph):
    colors = {} #初始化一个空的字典，用于存储每个顶点的颜色
    # 对图的每个顶点进行着色
    for vertex in graph:
        # 获取与当前顶点相邻的已经着色的顶点的颜色集合
        neighbor_colors = {colors[neighbor] for neighbor in graph[vertex] if neighbor in colors}
        # 找到一个可用的颜色，该颜色未被相邻的顶点使用
        available_colors = set(range(1, len(graph) + 1)) - neighbor_colors
        # 选择一个可用的颜色并将其分配给当前顶点
        chosen_color = min(available_colors)
        colors[vertex] = chosen_color
    return colors

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D', 'E'],
    'D': ['B', 'C', 'E'],
    'E': ['C', 'D']
}
colors_result = graph_coloring(graph)
print(colors_result)
