import numpy as np
from numpy import random
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文

def haversine_distance(coord1, coord2):
    # 将经纬度从度数转换为弧度
    lat1, lon1 = np.radians(coord1[1]), np.radians(coord1[0])
    lat2, lon2 = np.radians(coord2[1]), np.radians(coord2[0])
    # Haversine 公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    # 地球半径（单位：公里）
    radius = 6371.0
    # 计算距离
    distance = radius * c
    return distance
coordinates = np.array([
    [104.107895, 30.457623],
    [104.642054, 30.702534],
    [104.055628, 30.527435],
    [103.721549, 30.931824],
    [104.112324, 30.668907],
    [104.077094, 30.674277],
    [104.214562, 30.531782],
    [103.943217, 30.782159],
    [104.145552, 30.605267],
    [104.028378, 30.676363],
    [103.974325, 30.735671],
    [104.076092, 30.624577],
    [104.060045, 30.633286],
    [104.042753, 30.637203],
    [104.034653, 30.525472],
    [103.922376, 30.570071],
    [103.693276, 30.632171],
    [104.101776, 30.632171],
    [104.234756, 30.568442],
    [104.074576, 30.624871],
    [104.204776, 30.518071]
])
# 初始化距离矩阵
num_points = len(coordinates)
dist_matrix = np.zeros((num_points, num_points))
# 计算距离矩阵
for i in range(num_points):
    for j in range(num_points):
        dist_matrix[i, j] = haversine_distance(coordinates[i], coordinates[j])
values = [0, 1.151, 1.273, 1.512, 2.311, 0.827, 4.185, 2.342, 1.124, 1.367, 4.227,
          0.622, 0.937, 1.722, 3.075, 1.252, 5.221, 2.879, 2.357, 4.212, 3.661]
# 46t

def xuhao(x,v):  # 获取x中第v个1的纵坐标
    vv = 0
    for i in range(len(x)):
        if x[i] == 1:
            vv += 1
            if vv == v:
                return i
def dfs(graph, start, end, k, matrix):
    xx = []
    dd = [0]*k
    for i in range(k):
        xxx = [start]
        down = xuhao(graph[start],i+1)
        dd[i] += float(matrix[start,down])
        start = down
        xxx.append(start)
        while start != end:
            down = xuhao(graph[start],1)
            dd[i] += matrix[start][down]
            xxx.append(down)
            start = down
        xx.append(xxx)
    vv = [0]*k
    for i in range(len(xx)):
        for j in range(1,len(xx[i])-1):
            vv[i] += values[xx[i][j]]
    return xx,dd,vv


def xijdef(x):
    xij = np.zeros((21,21), dtype=int)
    # 将主对角线上面的一条线设为1
    for i in range(1,21-x):
        xij[i, i+x] = 1
    for i in range(1,x+1):
        xij[-i,0] = 1
    for i in range(1,x+1):
        xij[0,i] = 1
    return xij

def fun(X):  # 目标函数和约束条件
    x = X.flatten() #将X变为一维数组
    xij = xijdef(int(x[-1]))
    for i in range(15):
        xij[:,int(x[2*i])] , xij[:,int(x[2*i+1])] = xij[:,int(x[2*i+1])].copy() , xij[:,int(x[2*i])].copy()
    for i in range(15,30):
        xij[int(x[2*i]),:] , xij[int(x[2*i+1]),:] = xij[int(x[2*i+1]),:].copy() , xij[int(x[2*i]),:].copy()
    result,dd,vv = dfs(xij,0,0,int(x[-1]),dist_matrix)  # 获得路径与路程

    st = 0
    lis = [element for sublist in result for element in sublist if element != 0]
    if len(lis) != 20:
        st += 1000000000000
    if max(dd) > 150: # 最大距离限制
        st += 1000000000
    if max(vv) > 10: # 最大容量限制
        st += 1000000000
    for i in range(1,21): # 每个点都要走
        if sum(xij[:,i]) != 1:
            st += 1000000000
        if sum(xij[i,:]) != 1:
            st += 1000000000

    for i in range(21):  # 避免车辆来回跑
        for j in range(i+1):
            if xij[i,j] == xij[j,i] and xij[i,j] == 1:
                st += 1000000000

    fx1 = x[-1]*150  # 车辆固定成本
    fx2 = 0
    fx3 = 0
    e = np.e
    for i in range(len(result)):
        ddd = 0
        for j in range(1,len(result[i])):
            ddd += dist_matrix[result[i][j-1],result[i][j]]
            if j == len(result[i])-1:
                fx3 += 6*dist_matrix[result[i][j-1],result[i][j]]
            else:
                fx2 += (1-1/e**(0.05*(ddd/23)))*500*values[result[i][j]] # 货损成本
                fx3 += ddd*7*values[result[i][j]] # 运输费用
    return fx1+fx2+fx3+st


def dd2(best_x, x):  #欧氏距离
    best_x = np.array(best_x)   #转化成numpy数组
    x = np.array(x)          #转化成numpy数组
    c = np.sum(pow(x - best_x, 2), axis=1)    #求方差，在行上的标准差
    d = pow(c, 0.5)   #标准差
    return d
def new_min(arr):  #求最小
    min_data = min(arr)   #找到最小值
    key = np.argmin(arr)  #找到最小值的索引
    return min_data, key
def type_x(xx,type,n):  #变量范围约束
    for v in range(n):
        if type[v] == -1:
            xx[v] = np.maximum(sub[v], xx[v])
            xx[v] = np.minimum(up[v], xx[v])
        elif type[v] == 0:
            xx[v] = np.maximum(sub[v], int(xx[v]))
            xx[v] = np.minimum(up[v], int(xx[v]))
        else:
            xx[v] = np.maximum(sub[v], random.randint(0,2))
            xx[v] = np.minimum(up[v], random.randint(0,2))
    return xx
def woa(sub,up,type,nums,det):
    n = len(sub)  # 自变量个数
    num = nums * n  # 种群大小
    x = np.zeros([num, n])  #生成保存解的矩阵

    f = np.zeros(num)   #生成保存值的矩阵
    for s in range(num):      #随机生成初始解
        for v in range(n):
            rand_data = np.random.uniform(0,1)
            x[s, v] = sub[v] + (up[v] - sub[v]) * rand_data
        x[s, :] = type_x(x[s, :],type,n)
        f[s] = fun(x[s, :])
    best_f, a = new_min(f)  # 记录历史最优值
    best_x = x[a, :]  # 记录历史最优解
    trace = np.array([deepcopy(best_f)]) #记录初始最优值,以便后期添加最优值画图
    ############################ 改进的鲸鱼算法 ################################
    xx = np.zeros([num, n])
    ff = np.zeros(num)
    Mc = (up - sub) * 0.1  # 猎物行动最大范围
    for ii in tqdm(range(det)):      #设置迭代次数，进入迭代过程
        # 猎物躲避,蒙特卡洛模拟，并选择最佳的点作为下一逃跑点 #########！！！创新点
        d = dd2(best_x, x)  #记录当前解与最优解的距离
        d.sort()  #从小到大排序,d[0]恒为0
        z = np.exp(-d[1] / np.mean(Mc))  # 猎物急躁系数
        z = max(z, 0.1)     #决定最终系数
        yx = []  #初始化存储函数值
        dx = []  #初始化存储解
        random_rand = random.random() #0-1的随机数
        for i in range(30):    #蒙特卡洛模拟的次数
            m = [random.choice([-1, 1]) for _ in range(n)] #随机的-1和1
            asd = best_x + Mc * z * ((det-ii )/det) * random_rand * m   #最优解更新公式
            xd = type_x(asd,type,n)  #对自变量进行限制
            if i < 1:
                dx = deepcopy(xd)
            else:
                dx = np.vstack((dx,xd))   #存储每一次的解
            yx=np.hstack((yx,fun(xd)))    #存储每一次的值
        best_t, a = new_min(yx)  # 选择最佳逃跑点
        best_c = dx[a, :]   #最佳逃跑点
        if best_t < best_f:   #与鲸鱼算法得到的最优值对比
            best_f = best_t   #更新最优值
            best_x = best_c   #更新最优解
        ############################# 鲸鱼追捕 #################################
        w = (ii / det)**3   #自适应惯性权重!!!创新点
        a = (2 - 2*ii/det)*(1- w)  #a随迭代次数从2非线性下降至0！！！创新点
        pp=0.7 if ii <= 0.5*det else 0.4
        for i in range(num):
            r1 = np.random.rand()  # r1为[0,1]之间的随机数
            r2 = np.random.rand()  # r2为[0,1]之间的随机数
            A = 2 * a * r1 - a
            C = 2 * r2
            b = 1     #螺旋形状系数
            l = np.random.uniform(-1,1)  #参数l
            p = np.random.rand()
            if p < pp:
                if abs(A) >= 1:
                    rand_leader = np.random.randint(0, num)
                    X_rand = x[rand_leader, :]
                    D_X_rand = abs(C * X_rand - x[i, :])
                    xx[i, :] = w*X_rand - A * D_X_rand
                    xx[i, :] = type_x(xx[i, :],type,n) #对自变量进行限制
                elif abs(A) < 1:
                    D_Leader = abs(C * best_x - x[i, :])
                    xx[i, :] = w*best_x - A * D_Leader
                    xx[i, :] = type_x(xx[i, :],type,n) #对自变量进行限制
            elif p >= pp:
                D = abs(best_x - x[i, :])
                xx[i, :] = D*np.exp(b*l)*np.cos(2*np.pi*l) + (1-w)*best_x   #完整的气泡网捕食公式
                xx[i, :] = type_x(xx[i, :],type,n) #对自变量进行限制
            ff[i] = fun(xx[i, :])
            if len(np.unique(ff[:i]))/(i+1) <= 0.1:     #limit阈值 + 随机差分变异！！！创新点
                xx[i,:] = (r1*(best_x-xx[i,:]) +
                           r2*(x[np.random.randint(0,num),:] - xx[i,:]))
                xx[i, :] = type_x(xx[i, :],type,n) #对自变量进行限制
                ff[i] = fun(xx[i, :])
        #将上一代种群与这一代种群以及最优种群结合，选取排名靠前的个体组成新的种群
        F = np.hstack((np.array([best_f]), f, ff))
        F, b = np.sort(F,axis=-1,kind='stable'), np.argsort(F)#按小到大排序,获得靠前的位置
        X = np.vstack(([best_x], x, xx))[b, :]
        f = F[:num]  #新种群的位置
        x = X[:num, :]  #新种群的位置
        best_f, a = new_min(f)  # 记录历史最优值
        best_x = x[a , :]  # 记录历史最优解
        trace = np.hstack((trace, [best_f]))
    return best_x,best_f,trace


s = np.zeros((1,60))
sub = np.concatenate((s+1,np.array([[5]])), axis=1).ravel()  # 自变量下限
up = np.concatenate((s+20,np.array([[7]])), axis=1).ravel()  # 自变量上限
type = np.concatenate((s,np.array([[0]])), axis=1).ravel()    #-1是有理数，0是整数，1是0-1变量
best_x,best_f,trace = woa(sub,up,type,40,10)     #种群大小，迭代次数
#种群大小可以为自变量个数，迭代次数看情况
print('最优解为：')
print(best_x)
print('最优值为：')
print(float(best_f))

xij = xijdef(int(best_x[-1]))
for i in range(15):
    xij[:,int(best_x[2*i])] , xij[:,int(best_x[2*i+1])] = xij[:,int(best_x[2*i+1])].copy() , xij[:,int(best_x[2*i])].copy()
for i in range(15,30):
    xij[int(best_x[2*i])] , xij[int(best_x[2*i+1])] = xij[int(best_x[2*i+1])].copy() , xij[int(best_x[2*i])].copy()
result,dd,vv = dfs(xij,0,0,int(best_x[-1]),dist_matrix)
print(result)
print(dd)
print(vv)
fx1 = best_x[-1]*150  # 车辆固定成本
fx2 = 0
fx3 = 0
e = np.e
for i in range(len(result)):
    ddd = 0
    for j in range(1,len(result[i])):
        ddd += dist_matrix[result[i][j-1],result[i][j]]
        if j == len(result[i])-1:
            fx3 += 4*dist_matrix[result[i][j-1],result[i][j]]
        else:
            fx2 += (1-1/e**(0.05*(ddd/23)))*500*values[result[i][j]] # 货损成本
            fx3 += ddd*7*values[result[i][j]] # 运输费用
print(fx1,fx2,fx3)

plt.title('鲸鱼算法')
plt.plot(range(1,len(trace)+1),trace, color='r')
plt.show()
