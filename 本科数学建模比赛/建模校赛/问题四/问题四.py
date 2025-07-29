import ydata_profiling as pp
import webbrowser
import warnings
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer,KNNImputer
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pylab
import networkx as nx
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

##########################     运输量数据      #################################
df = pd.read_excel('附件2(Attachment 2)2023-51MCM-Problem B.xlsx',header = 0)
list = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
def yun(day):
    index = df[(df['日期(年/月/日) (Date Y/M/D)'] == f'2023-04-{day}')].index.tolist()
    df1=df.iloc[index,:].copy()
    yun = np.zeros((25, 25))
    for i in range(len(list)):
        for v in range(len(list)):
            index0 = df1[(df1['发货城市 (Delivering city)'] == list[i])&(df1['收货城市 (Receiving city)'] == list[v])].index.tolist()
            if len(index0)==0:
                zhi = 0
                yun[i, v] = np.array(zhi)
            else:
                zhi = df.iloc[index0,:]['快递运输数量(件) (Express delivery quantity (PCS))']
                yun[i,v] = np.array(zhi)
    return yun
yun23 = yun(23)
yun24 = yun(24)
yun25 = yun(25)
yun26 = yun(26)
yun27 = yun(27)
#print('23号运量',yun23)
##############################    固定成本数据    #############################
df4 = pd.read_excel('附件3(Attachment 3)2023-51MCM-Problem B.xlsx',header = 0)
graph = np.zeros((25, 25))
for i in range(len(list)):
    for v in range(len(list)):
        index = df4[(df4['起点 (Start)'] == list[i])&(df4['终点 (End)'] == list[v])].index.tolist()
        if len(index)==0:
            zhi = 0
            graph[i, v] = np.array(zhi)
        else:
            zhi = df4.iloc[index,:]['固定成本 (Fixed cost)']
            graph[i,v] = np.array(zhi)
#print(graph)
def floyd_warshall(graph):
    n = len(graph)
    next_nodes = [[None]*n for _ in range(n)] # 用于记录下一个节点
    dp = [[float('inf')]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dp[i][j] = 0
            elif graph[i][j] != 0:
                dp[i][j] = graph[i][j]
                next_nodes[i][j] = j # 如果两点直接有边，则下一个节点为j
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dp[i][j] > dp[i][k]+dp[k][j]:
                    dp[i][j] = dp[i][k]+dp[k][j]
                    next_nodes[i][j] = next_nodes[i][k] # 更新下一个节点
    dp = np.round(dp, 1)
    return dp, next_nodes
def get_path(next_nodes, i, j):
    if next_nodes[i][j] is None:
        return []
    path = [i]
    while i != j:
        i = next_nodes[i][j]
        path.append(i)
    return path
dp, next_nodes = floyd_warshall(graph)
#print(dp)  #最小成本矩阵#########################################################
p = []
new_lu = []
for i in range(len(list)):
    for v in range(len(list)):
        if yun23[i][v] != 0:
            lu = get_path(next_nodes, i, v)
            p.append(lu)  # 路径，不过是数字的

            replace_dict = {i: chr(65+i) for i in range(25)}
            new_lu.append([replace_dict.get(x, x) for x in lu])
#print(new_lu) #输出最短路径
'''
xian = []
new_xian = []
for i in range(len(p)):
    xian.append([p[i][0],p[i][-1]])
    replace_dict = {i: chr(65 + i) for i in range(25)}
    new_xian.append([replace_dict.get(x, x) for x in xian[i]])
'''

new_yun23 = [[i] for i in np.array(yun23).ravel() if i != 0]

#for i in range(len(p)):
result = [[new_yun23[i],new_lu[i]] for i in range(len(p))]
#print(result)
'''
result_z = []
for i in range(len(list)):
    for v in range(len(list)):
        for u in range(len(result)):
            lst = result[u][1]
            a = f'{list[i]}'
            b = f'{list[v]}'
            pattern = fr'{a}{b}'
            match = re.search(pattern, ''.join(lst))
            if match:
                result_z.append([f'{a}-->{b}',result[u][0][0]])
#print(result_z)
i = 0
while i < len(result_z)-1:
    if result_z[i][0] == result_z[i+1][0]:
        result_z[i][1] = result_z[i][1]+result_z[i+1][1]
        result_z.remove(result_z[i+1])
        #print(result_z)
    else:
        i += 1
print(result_z)
result_x = []
for i in range(len(list)):
    for v in range(len(list)):
        if graph[i][v] != 0:
            result_x.append([f'{list[i]}-->{list[v]}',graph[i][v]])
print(result_x)
v = 0
zc = 0
for i in range(len(result_z)):
    while v <= len(result_x):
        if result_z[i][0] == result_x[v][0]:
            zc += result_x[v][1] * (1 + (result_z[i][1]/200)**3)
            v = 0
            break
        else:
            v += 1
print(zc)'''
###########################  计算总成本   #################################

s = 0
for i in range(len(list)):
    for v in range(len(list)):
        s += dp[i][v]*(1+(yun23[i][v]/200))**3
print('23号：',s)
s = 0
for i in range(len(list)):
    for v in range(len(list)):
        s += dp[i][v]*(1+(yun24[i][v]/200))**3
print('24号：',s)
s = 0
for i in range(len(list)):
    for v in range(len(list)):
        s += dp[i][v]*(1+(yun25[i][v]/200))**3
print('25号：',s)
s = 0
for i in range(len(list)):
    for v in range(len(list)):
        s += dp[i][v]*(1+(yun26[i][v]/200))**3
print('26号：',s)
s = 0
for i in range(len(list)):
    for v in range(len(list)):
        s += dp[i][v]*(1+(yun27[i][v]/200))**3
print('27号：',s)

matrix = graph
# 创建一个无向图对象
G = nx.Graph()
# 添加节点
for i in range(len(matrix)):
    G.add_node(i)
# 添加边
for i in range(len(matrix)):
    for j in range(i+1, len(matrix)):
        if matrix[i][j] != 0:
            G.add_edge(i, j, weight=matrix[i][j])
c = []
for i in range(len(p)):
    c += [(p[i][v], p[i][v+1]) for v in range(len(p[i])-1)]

# 绘制无向图
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v):G[u][v]['weight'] for u, v in G.edges})
nx.draw_networkx_labels(G, pos, labels={i:chr(65+i) for i in range(len(matrix))})
nx.draw_networkx_edges(G, pos, edgelist=c, edge_color='blue', width=2)
plt.show()



'''
[[0, 6, 2, 5, 8, 6], 
[6, 0, 5, 10, 3, 9], 
[1, 5, 0, 5, 6, 4],
[5, 10, 5, 0, 8, 2], 
[7, 3, 6, 8, 0, 6], 
[5, 9, 4, 2, 6, 0]]
    A    B    C    D    E    F    G    H    I    J    K     L   M    N    O     P    Q     R    S    T    U    V    W   X   Y
A[[ 0.  17.6  2.4 17.4  6.  10.2  9.  16.4 15.  14.6 11.6  3.6 13.8  6.  11.4  8.4  9.  13.2 18.6  3.6  9.   9.6  7.8 13.8 21. ]
B [16.8  0.  17.4  9.6 10.8  9.6  9.6  3.6  3.6  3.   4.8 16.2  6.6 13.2   8.4  9.6 11.4  6.8  7.2 13.2 10.8  7.8 11.4  6.   9.6]
C [ 2.4 17.   0.  18.   6.6 10.8  9.6 15.8 15.6 14.  11.   1.2 14.4  3.6  12.   7.8  6.6 13.  19.2  4.8  9.6  9.   8.4 14.4 21.6]
D [16.   7.2 16.6  0.  10.   6.   6.  10.2  4.2  8.4  9.2 15.4  3.  12.4   4.8  9.  10.8  8.4  2.4 12.4  7.2  7.2  7.8  6.6  3.6]
E [ 6.  11.6  6.6 11.4  0.   4.2  3.  10.4  9.   8.6  5.6  5.4  7.8  2.4   5.4  2.4  4.2  7.2 12.6  2.4  3.   3.6  1.8  7.8 15. ]
F [10.2  8.  10.8  7.4  4.2  0.   2.6  7.4  5.   5.6  3.2  9.6  3.8  6.6   1.4  3.   4.8  3.   8.6  6.6  3.   1.2  4.4  3.8 11. ]
G [10.   9.  10.6  8.4  4.   2.4  0.   8.4  6.   6.6  5.6  9.4  4.8  6.4   2.4  3.2  5.   5.4  9.6  6.4  1.2  3.6  1.8  4.8 12. ]
H [16.8  4.2 17.4 13.8 10.8 10.2 10.2  0.   7.8  3.   4.8 16.2 10.8 13.2   9.   9.6 11.4  6.8 11.4 13.2 11.4  7.8 12.   6.6 13.8]
I [16.   3.  16.6  6.  10.   6.   6.   6.   0.   4.2  6.  15.4  3.  12.4   4.8  9.  10.8  4.2  3.6 12.4  7.2  7.2  7.8  2.4  6. ]
J [13.8  3.  14.4 12.   7.8  7.2  7.2  1.8  6.   0.   1.8 13.2  8.4 10.2   6.   6.6  8.4  3.8  9.6 10.2  8.4  4.8  9.   3.6 12. ]
K [12.   6.  12.6 12.8  6.   5.4  8.   4.8  6.8  3.   0.  11.4  9.2  8.4   6.8  4.8  6.6  2.  10.4  8.4  6.6  3.   7.8  4.4 12.8]
L [ 3.6 15.8  1.2 16.8  5.4  9.6  8.4 14.6 14.4 12.8  9.8  0.  13.2  2.4  10.8  6.6  5.4 11.8 18.   3.6  8.4  7.8  7.2 13.2 20.4]
M [13.   4.2 13.6  3.6  7.   3.   3.   7.2  1.2  5.4  6.2 12.4  0.   9.4   1.8  6.   7.8  5.4  4.8  9.4  4.2  4.2  4.8  3.6  7.2]
N [ 6.6 13.4  4.2 14.4  3.   7.2  6.  12.2 12.  10.4  7.4  3.  10.8  0.   8.4  4.2  3.   9.4 15.6  5.4  6.   5.4  4.8 10.8 18. ]
O [11.2  6.6 11.8  6.   5.2  1.2  1.2  6.   3.6  4.2  4.4 10.6  2.4  7.6   0.   4.2  6.   4.2  7.2  7.6  2.4  2.4  3.   2.4  9.6]
P [ 7.2  9.2  7.8 10.4  1.2  3.   4.2  8.   8.   6.2  3.2  6.6  6.8  3.6   4.4  0.   1.8  5.2 11.6  3.6  1.8  1.2  3.   6.8 14. ]
Q [ 8.4 10.4  6.6 11.6  2.4  4.2  5.4  9.2  9.2  7.4  4.4  5.4  8.   2.4   5.6  1.2  0.   6.4 12.8  4.8  3.   2.4  4.2  8.  15.2]
R [13.8  7.2 14.4 10.8  7.8  3.6  6.   6.   4.8  4.2  2.4 13.2  7.2 10.2   4.8  6.6  8.4  0.   8.4 10.2  6.6  4.8  7.8  2.4 10.8]
S [18.4  5.4 19.   2.4 12.4  8.4  8.4  8.4  2.4  6.6  8.4 17.8  5.4 14.8   7.2 11.4 13.2  6.6  0.  14.8  9.6  9.6 10.2  4.8  2.4]
T [ 3.6 14.   4.2 13.8  2.4  6.6  5.4 12.8 11.4 11.   8.   3.  10.2  4.8   7.8  4.8  6.6  9.6 15.   0.   5.4  6.   4.2 10.2 17.4]
U [ 9.2  9.2  9.8  8.6  3.2  1.2  3.   8.6  6.2  6.8  4.4  8.6  5.   5.6   2.6  2.   3.8  4.2  9.8  5.6  0.   2.4  1.8  5.  12.2]
V [ 9.   8.   9.6  9.8  3.   2.4  5.   6.8  7.4  5.   2.   8.4  6.2  5.4   3.8  1.8  3.6  4.  11.   5.4  3.6  0.   4.8  6.2 13.4]
W [ 8.2 10.2  8.8  9.6  2.2  2.4  1.2  9.6  7.2  7.8  5.6  7.6  6.   4.6   3.6  3.2  5.   5.4 10.8  4.6  1.2  3.6  0.   6.  13.2]
X [13.6  4.8 14.2  8.4  7.6  3.6  3.6  3.6  2.4  1.8  3.6 13.   4.8  10.   2.4  6.6  8.4  1.8  6.  10.   4.8  4.8  5.4  0.   8.4]
Y [18.4  7.8 19.   2.4 12.4  8.4  8.4 10.8  4.8  9.  10.8 17.8  5.4 14.8   7.2 11.4 13.2  9.   2.4 14.8  9.6  9.6 10.2  7.2  0. ]]
 
'''
'''
#print('--------------------发货----------------------')
falist = []
for i in range(len(list)):
    index1 = df1[(df1['发货城市 (Delivering city)'] == list[i])].index.tolist()
    df2 = df.iloc[index1,:].copy()
    df2 = df2['快递运输数量(件) (Express delivery quantity (PCS))'].sum()
    falist.append(df2)
    #print(list[i],df2)
#print('--------------------收货----------------------')
shoulist = []
for i in range(len(list)):
    index2 = df1[(df1['收货城市 (Receiving city)'] == list[i])].index.tolist()
    df3 = df.iloc[index2,:].copy()
    df3 = df3['快递运输数量(件) (Express delivery quantity (PCS))'].sum()
    shoulist.append(df3)
    #print(list1[i],df3)
list_0 = np.hstack((np.array(falist).reshape(-1,1),np.array(shoulist).reshape(-1,1)))
'''