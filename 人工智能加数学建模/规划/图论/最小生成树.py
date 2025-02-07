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
data0 = [[0, 6, 1, 5, 0, 0],
         [6, 0, 5, 0, 3, 0],
         [1, 5, 0, 5, 6, 4],
         [5, 0, 5, 0, 0, 2],
         [0, 3, 6, 0, 0, 6],
         [0, 0, 4, 2, 6, 0]]
data=DataFrame(data0)
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
start_node=1
# 目的地
target_node=6
# 设置各顶点坐标（方便绘图，非实际位置）
pos={1:(1,8),2:(4,10),3:(11,11),4:(14,8),5:(5,7),6:(10,6),7:(3,5),8:(6,4),9:(8,4),10:(14,5),11:(2,3),12:(5,1),13:(8,1),14:(13,3)}

# 创建空的无向图
G=nx.Graph()
# 给无向图的边赋予权值
G.add_weighted_edges_from(coo_list)

plt.figure()
plt.suptitle('最小生成树')
T=nx.minimum_spanning_tree(G,algorithm='kruskal')
# T=nx.minimum_spanning_tree(G,algorithm='prim')
# T=nx.minimum_spanning_tree(G,algorithm='boruvka')
ed = [list(i) for i in T.edges]
data0 = np.array(data0)
s_weight = 0
for i in range(len(ed)):
    wei = np.array(ed[i])
    weight = data0[wei[0]-1,wei[1]-1]
    s_weight += weight
    print(wei,weight)
print('总路程：',s_weight)

# 绘制无向加权图
nx.draw(G,pos,with_labels=True)
# 设置最小生成树的顶点颜色
nx.draw_networkx_nodes(G,pos,nodelist=T.nodes,node_color='yellow',edgecolors='red')
# 设置最小生成树的边颜色和宽度
nx.draw_networkx_edges(G,pos,edgelist=T.edges,edge_color='blue',width=2)
# 显示无向加权图的边的权值
labels=nx.get_edge_attributes(G,'weight')
# 显示边的权值
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,font_color='purple',font_size=10)
nx.draw_networkx_nodes(G,pos,node_color='#00ff00',edgecolors='red')
# 添加图注
plt.text(0.1, 0.1, "0:h\n1:g\n2:r", transform=plt.gca().transAxes, ha="center",fontsize=14)
plt.show()