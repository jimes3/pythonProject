import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from pandas.core.frame import DataFrame
import networkx as nx
import matplotlib.pyplot as plt
# 避免图片无法显示中文
plt.rcParams['font.sans-serif']=['SimHei']
# 显示所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# 读取数据
#data=pd.read_excel('data.xlsx',sheet_name='Sheet1',index_col=0)
data = [[0, 6, 1, 5, 0, 0],
         [6, 0, 5, 0, 3, 0],
         [1, 5, 0, 5, 6, 4],
         [5, 0, 5, 0, 0, 2],
         [0, 3, 6, 0, 0, 6],
         [0, 0, 4, 2, 6, 0]]
data=DataFrame(data)#这时候是以行为标准写入的
#data=data.T#转置之后得到想要的结果
#data.rename(index={0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g'},
            #columns={0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g'},inplace=True)#注意这里0和1都不是字符串
data=data.fillna(0)
#print('矩阵的空值以0填充：\n',data)
coo=coo_matrix(np.array(data))
# 矩阵行列的索引默认从0开始改成从1开始
coo.row+=1
coo.col+=1
data=[int(i) for i in coo.data]
coo_tuple=list(zip(coo.row,coo.col,data))
coo_list=[]
for i in coo_tuple:
    coo_list.append(list(i))
#print(coo_list)
# 出发点
start_node=5
# 目的地
target_node=1
# 设置各顶点坐标（只是方便绘图，并不是实际位置）
pos={1:(1,8),2:(4,10),3:(11,11),4:(14,8),5:(5,7),6:(10,6),7:(3,5),8:(6,4),9:(8,4),10:(14,5),11:(2,3),12:(5,1),13:(8,1),14:(13,3)}

# 创建空的无向图
G=nx.DiGraph()
# 给无向图的边赋予权值
G.add_weighted_edges_from(coo_list)

# 单源最短路径dijkstra迪杰斯特拉算法求最短路径和最短路程
min_path_dijkstra=nx.dijkstra_path(G,source=start_node,target=target_node)
min_path_length_dijkstra=nx.dijkstra_path_length(G,source=start_node,target=target_node)
print('\ndijkstra算法：顶点%d到顶点%d的最短路径:%s，最短路程:%d'%(start_node,target_node,min_path_dijkstra,min_path_length_dijkstra))

# 多源最短路径floyd弗洛伊德算法求最短路径和最短路程
index_floyd=list(nx.floyd_warshall(G,weight='weight'))
min_path_length_floyd=nx.floyd_warshall_numpy(G,weight='weight')
min_path_length_floyd=pd.DataFrame(data=min_path_length_floyd,index=index_floyd,columns=index_floyd)
print('\nfloyd算法：各顶点之间的最短路程矩阵：\n',min_path_length_floyd)

plt.figure()
plt.suptitle('最短路径图')
# 绘制无向加权图
nx.draw(G,pos,with_labels=True)
# 显示无向加权图的边的权值
labels=nx.get_edge_attributes(G,'weight')

# 设置顶点颜色
nx.draw_networkx_nodes(G,pos,node_color='yellow',edgecolors='red')
# 设置边颜色和宽度
edge = []
for i in range(0,len(min_path_dijkstra)-1):
    a = min_path_dijkstra[i]
    b = min_path_dijkstra[i+1]
    c = (a,b)
    edge.append(c)
nx.draw_networkx_edges(G,pos,edgelist=edge,edge_color='blue',width=2)
# 显示边的权值
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,font_color='purple',font_size=10)
nx.draw_networkx_nodes(G,pos,nodelist=min_path_dijkstra,node_color='#00ff00',edgecolors='red')#最短路径标红
plt.show()