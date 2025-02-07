import numpy as np
from numpy import random
import pandas as pd
from copy import deepcopy
from scipy.optimize import least_squares

df = np.array(pd.read_excel('地理表.xlsx',header = None))
def ju_mian(df):  # 拟合海底坡面
    X = np.linspace(0, 0.02*1852*(len(df[0]-1)),len(df[0]))
    Y = np.linspace(0, 0.02*1852*(len(df[0])),len(df))
    Z = -df
    X, Y = np.meshgrid(X, Y)
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    def plane_fit(coefficients, X, Y, Z):
        a, b, c, d = coefficients
        return a * X + b * Y + c * Z + d
    initial_guess = [1.0, 1.0, 1.0, 1.0]
    result = least_squares(plane_fit, initial_guess, args=(X, Y, Z))
    a, b, c, d = result.x
    return a,b,c,d
def yuansu(df): # 求解覆盖宽度
    a,b,c,d = ju_mian(df)  # 坡面方程的系数
    fa1 = np.array([a,b,c]) # 海底拟合坡面的法向量

    xian = np.array([0,1,0])  # 测线的方向向量(投影)

    touying1 = (np.dot(xian, fa1) / np.linalg.norm(fa1)) * fa1 # 坡面上的投影
    co_si = (np.dot(touying1, xian) /  # 计算两个投影向量之间的夹角（弧度）
             (np.linalg.norm(touying1) * np.linalg.norm(xian)))
    angle_r = np.arccos(co_si)
    angle_deg2 = 90-np.degrees(angle_r)  # 将弧度转换为度数
    #print(f"测线方向为{theta}度时，侧线垂面坡度为：{round(angle_deg2,4)}°")

    D = np.mean(df)  # 求得深度
    g3 = np.sqrt(3)
    s2 = np.sin(np.deg2rad(30-angle_deg2))
    s3 = np.sin(np.deg2rad(30+angle_deg2))
    c1 = np.cos(np.deg2rad(angle_deg2))
    w1 = g3/(2*s2)*c1*D  # 坡度较大则小于0
    w2 = g3/(2*s3)*c1*D
    w = w1+w2
    return w,w1,w2


def fun(X,www):  # 目标函数和约束条件
    x = X.flatten() #将X变为一维数组
    ww = []
    shuti = 1 + (36*(x[0]-45))/1610
    for i in range(1,6):
        data = np.array([row[int(max(0, x[0]-4)) : int(min(len(df[0]), x[0] +4+1))]
                         for row in df[int(max(0, x[i]*shuti-4)) : int(min(len(df), x[i]*shuti +4+1))]]) # 前x后y
        w,w1,w2 = yuansu(data)
        ww .append(w)
    re = (x[0]-45)*37.04-0.5*np.mean(ww)-0.9*sum(www)
    if re > 0:
        return 10000
    else:
        return abs(re)

s = np.zeros((1,6))
sub = np.array([45,0,0,0,0,0]).ravel()  # 自变量下限
up = np.array([160,70,70,70,70,70]).ravel()  # 自变量上限
type = np.array(s-1).ravel()    #-1是有理数，0是整数，1是0-1变量

def dd2(best_x, x):  #欧氏距离
    best_x = np.array(best_x)
    x = np.array(x)
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
def woa(sub,up,type,nums,det,www):
    n = len(sub)  # 自变量个数
    num = nums * n  # 种群大小
    x = np.zeros([num, n])  #生成保存解的矩阵
    f = np.zeros(num)   #生成保存值的矩阵
    for s in range(num):      #随机生成初始解
        rand_data = np.random.uniform(0,1)
        x[s, :] = sub + (up - sub) * rand_data
        x[s, :] = type_x(x[s, :],type,n)
        f[s] = fun(x[s, :],www)
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
        for i in range(10):    #蒙特卡洛模拟的次数
            m = [random.choice([-1, 1]) for _ in range(n)] #随机的-1和1
            asd = best_x + Mc * z * ((det-ii )/det) * random_rand * m   #最优解更新公式
            xd = type_x(asd,type,n)  #对自变量进行限制
            if i < 1:
                dx = deepcopy(xd)
            else:
                dx = np.vstack((dx,xd))   #存储每一次的解
            yx=np.hstack((yx,fun(xd,www)))    #存储每一次的值
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
            ff[i] = fun(xx[i, :],www)
            if len(np.unique(ff[:i]))/(i+1) <= 0.1:     #limit阈值 + 随机差分变异！！！创新点
                xx[i,:] = (r1*(best_x-xx[i,:]) +
                           r2*(x[np.random.randint(0,num),:] - xx[i,:]))
                xx[i, :] = type_x(xx[i, :],type,n) #对自变量进行限制
                ff[i] = fun(xx[i, :],www)
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

bx = []
bf = []
www = [0]
while True:
    best_x,best_f,trace = woa(sub,up,type,3,30,www)     #种群大小，迭代次数
    if best_x[0] >= 160:
        break
    #种群大小可以为自变量个数，迭代次数看情况
    print('最优解为：')
    print(best_x[0]*37.04) # 转化为m
    print('最优值为：')
    print(float(best_f*37.04))
    kuan = 2*(best_f + best_x[0]*37.04 - 0.9*sum(www))
    www.append(kuan)
    print('覆盖宽度：',kuan)
    bx.append(best_x[0]*37.04)
    bf.append(best_f*37.04)

print(bx)
print(bf)
print(www[1:])

'''
[1705.088731086144, 4842.21912220321]
[28.65037858887025, 40.634963444336684]
[3411.724458856352, 3545.52833096198]
'''