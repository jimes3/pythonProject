import numpy as np
import pandas as pd
from numpy import random
from copy import deepcopy
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

df = pd.read_excel("附件1 近5年402家供应商的相关数据.xlsx",sheet_name = '供应商的供货量（m³）')
df = df.iloc[[360, 228, 139, 107, 281, 329, 339, 274, 138, 355, 307, 97, 328, 130, 305, 142, 351,
              267, 193, 39, 75, 243, 364, 366, 66, 306, 246, 283, 30, 217, 54, 293, 79, 122, 373,
              345, 6, 363, 265, 113, 313, 149, 337, 150, 85, 290, 212, 128, 2, 145]]
#df.to_excel('前50名供应商.xlsx', index=False)
########################################  平均损耗率
data = pd.read_excel("附件2 近5年8家转运商的相关数据.xlsx",sheet_name = '运输损耗率（%）')
data = data.drop('转运商ID', axis=1)
data = np.array(data)
av = np.mean(data)
r = 0
for i in range(len(data)):
    r += np.sum(data[i] == 0)
shape = len(data)*len(data[0])
rant = av*shape/(shape-r)
#print(rant)
##############################################  平均供货量
x_df = np.array(pd.read_excel("问题一数据.xlsx",sheet_name = '供应商的供货量（m³）'))
av_up = []
for i in range(402):
    av2 = np.average(np.sort(x_df[i])[120:])
    av_up.append(round(av2,3))
avv = pd.DataFrame(av_up)
res = avv.iloc[[360, 228, 139, 107, 281, 329, 339, 274, 138, 355, 307, 97, 328, 130, 305, 142, 351, 267, 193, 39, 75,
                243, 364, 366, 66, 306, 246, 283, 30, 217, 54, 293, 79, 122, 373, 345, 6, 363, 265, 113, 313, 149, 337,
                150, 85, 290, 212, 128, 2, 145]]
aver = np.array(res).ravel()
############################################   c
def func(x):
    if x == 'A':
        x = 0.6
    elif x == 'B':
        x = 0.66
    elif x == 'C':
        x = 0.72
    return x
df['材料分类'] = df['材料分类'].apply(func)
def fun(X):  # 目标函数和约束条件
    x = X.flatten() #将X变为一维数组
    s = [0.003886, 0.003828, 0.003789, 0.00375, 0.003708, 0.003705, 0.003701, 0.003693, 0.003684,
         0.003681, 0.003679, 0.003676, 0.003676, 0.003663, 0.003655, 0.003644, 0.003638, 0.003628,
         0.003623, 0.003599, 0.003587, 0.003584, 0.00358, 0.003579, 0.003577, 0.003572, 0.003571,
         0.003565, 0.003564, 0.003562, 0.003559, 0.003556, 0.003553, 0.003546, 0.003544, 0.003539,
         0.003536, 0.003536, 0.003514, 0.00351, 0.003503, 0.003499, 0.003492, 0.003486, 0.003466,
         0.003463, 0.003461, 0.003431, 0.003382, 0.003352]
    f1 = 0
    f2 = 0
    for i in range(50):   #目标函数
        a = x[i]
        b = x[i]*s[i]
        f1 += a
        f2 += b
    st = -28200
    for i in range(50):
        st += x[i]*(1-0.0133)*aver[i]/df['材料分类'].iloc[i]
    return f1/f2 - st

s = np.zeros((1,50))
sub = np.array(s).ravel()  # 自变量下限
up = np.array(s+2).ravel()  # 自变量上限
type = (s+1).ravel()    #-1是有理数，0是整数，1是0-1变量

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
        rand_data = np.random.uniform(0,1)
        x[s, :] = sub + (up - sub) * rand_data
        x[s, :] = type_x(x[s, :],type,n)
        f[s] = fun(x[s, :])
    best_f, a = new_min(f)  # 记录历史最优值
    best_x = x[a, :]  # 记录历史最优解
    trace = np.array([deepcopy(best_f)]) #记录初始最优值,以便后期添加最优值画图
    ############################ 改进的鲸鱼算法 ################################
    xx = np.zeros([num, n])
    ff = np.zeros(num)
    Mc = (up - sub) * 0.1  # 猎物行动最大范围
    for ii in range(det):      #设置迭代次数，进入迭代过程
        # 猎物躲避,蒙特卡洛模拟，并选择最佳的点作为下一逃跑点 #########！！！创新点
        d = dd2(best_x, x)  #记录当前解与最优解的距离
        d.sort()  #从小到大排序,d[0]恒为0
        z = np.exp(-d[1] / np.mean(Mc))  # 猎物急躁系数
        z = max(z, 0.1)     #决定最终系数
        yx = []  #初始化存储函数值
        dx = []  #初始化存储解
        random_rand = random.random(n) #0-1的随机数
        for i in range(100):    #蒙特卡洛模拟的次数
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
        if ii % 10 == 0:
            print('迭代次数：',ii)
    return best_x,best_f,trace

best_x,best_f,trace = woa(sub,up,type,20,60)     #种群大小，迭代次数
print('最优解为：')
print(best_x)
print('最优值为：')
print(float(best_f))

plt.title('改进的鲸鱼算法')
plt.plot(range(1,len(trace)+1),trace, color='r')
plt.xlabel('迭代次数')
plt.show()
'''
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 0. 1. 1. 1. 1. 0. 1. 1. 0.
 0. 1. 0. 0. 1. 1. 1. 0. 0. 0. 1. 1. 0. 1. 0. 1. 1. 1. 0. 1. 0. 1. 0. 0.
 1. 0.]
 '''
