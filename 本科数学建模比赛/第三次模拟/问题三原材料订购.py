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

df = pd.read_excel("33家供应商.xlsx",sheet_name = 'Sheet1')
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
x_df = np.array(pd.read_excel("33家供应商.xlsx",sheet_name = 'Sheet1').drop(['供应商ID','材料分类'], axis=1))
av_up = []
for i in range(33):
    av2 = np.average(x_df[i][120:])
    av_up.append(round(av2,3))
aver = np.array(av_up).ravel()
############################################   c
def func(x):
    if x == 'A':
        x = [0.6,1.2,0.5]
    elif x == 'B':
        x = [0.66,1.1,1]
    elif x == 'C':
        x = [0.72,1,1.5]
    return x
df['材料分类'] = df['材料分类'].apply(func)

s = np.zeros((1,33))
sub = np.array(s).ravel()  # 自变量下限
up = np.array(s+3).ravel()  # 自变量上限
type = (s-1).ravel()    #-1是有理数，0是整数，1是0-1变量

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
def woa(sub,up,type,nums,det,stt=-28200):
    def fun(X,st=stt):  # 目标函数和约束条件
        x = X.flatten() #将X变为一维数组
        f1 = 0
        for i in range(33):   #目标函数
            f1 += int(x[i]*aver[i]*df['材料分类'].iloc[i][2])
        for i in range(33):
            st += x[i]*(1-0.0133)*aver[i]/df['材料分类'].iloc[i][0]
        if st <= 0:
            st = 100000000000
        return f1 + st
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
result = []
xi = np.array(df['材料分类'])
ress = []
best_x,best_f,trace = woa(sub,up,type,20,50)
liang = np.rint(best_x*aver)
result.append(liang)
res = 0
for i in range(33):
    res += liang[i]/xi[i][0]
print(int(res)>=28200)
print(int(res))
ress.append(int(res))
print('第1周:\n',liang)

for i in range(2,25):
    best_x,best_f,trace = woa(sub,up,type,20,50,stt=int(res)-56400)
    liang = np.rint(best_x*aver)
    result.append(liang)
    res = 0
    for v in range(33):
        res += liang[v]/xi[v][0]
    print(int(res)>=28200)
    print(int(res))
    ress.append(int(res))
    print(f'第{i}周:\n',liang)
print(result)
print(ress)