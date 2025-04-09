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
    lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
    lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])
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
    [33.052466,107.078539],#配送点
    [33.076266, 107.060763], [32.853819, 107.11977], [32.855425, 107.02023], [32.476447, 107.24196], [33.413498, 107.192686], [33.544629, 107.82695], [33.54876, 107.993614], [33.45927, 107.97293], [33.47337, 108.11846]
])
# 初始化距离矩阵
num_points = len(coordinates)
dist_matrix = np.zeros((num_points, num_points))
# 计算距离矩阵
for i in range(num_points):
    for j in range(num_points):
        dist_matrix[i, j] = haversine_distance(coordinates[i], coordinates[j])
values = [0, 8.28, 1.12, 0.49, 0.36, 0.52, 0.23, 0.24, 0.26, 0.23]


def fun(X):  # 目标函数和约束条件
    x = X.flatten() #将X变为一维数组
    l = len(x)
    point=[1,2,3,4,5,6,7,8,9]
    trace = []
    for i in range(l):
        trace.append(point[int(x[i])])
        del point[int(x[i])]
    lenth = dist_matrix[0,trace[0]]
    result = values[trace[0]]*dist_matrix[0,trace[0]]
    for i in range(1,l-1):
        lenth += dist_matrix[trace[i],trace[i+1]]
        result += values[trace[i+1]]*lenth
    return result/sum(values)


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


sub = np.array([0,0,0,0,0,0,0,0,0])  # 自变量下限
up = np.array([8,7,6,5,4,3,2,1,0])  # 自变量上限
type = np.array([0,0,0,0,0,0,0,0,0])    #-1是有理数，0是整数，1是0-1变量
best_x,best_f,trace = woa(sub,up,type,40,500)     #种群大小，迭代次数
#种群大小可以为自变量个数，迭代次数看情况
print('最优解为：')
point=[1,2,3,4,5,6,7,8,9]
trace1 = []
for i in range(len(best_x)):
    trace1.append(point[int(best_x[i])])
    del point[int(best_x[i])]
print(np.array(trace1))
print('最优值为：')
print(float(best_f))


plt.title('鲸鱼算法')
plt.plot(range(1,len(trace)+1),trace, color='r')
plt.show()
