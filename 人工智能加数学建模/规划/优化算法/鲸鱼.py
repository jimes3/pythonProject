import numpy as np
from numpy import random
from copy import deepcopy
import copy


def fun(data_list):
    arr = np.array(data_list)
    f = sum(pow(arr, 2))
    return f

def jingyu(X, K):
    n = len(X)
    Y = []
    y = np.zeros(n - 1)
    costheta = np.zeros(n - 1)
    theta = np.zeros(n - 1)
    k = np.zeros(n - 1)

    for i in range(n - 1):
        Y = np.append(X[0:i + 1], 0)
        y[i] = np.sqrt(np.sum((X[0:i + 2]) ** 2))
        # 计算角度（圈数）
        costheta[i] = (X[0:i + 2] @ Y) / (np.sqrt(np.sum(X[0:i + 2] ** 2)) * np.sqrt(np.sum(Y ** 2)))

        if np.isnan(costheta[i]) == 1:
            costheta[i] = 1
        if X[i + 1] >= 0:
            theta[i] = np.arccos(costheta[i]) / (2 * np.pi)
        else:
            theta[i] = 2 - np.arccos(costheta[i]) / (2 * np.pi)

        theta[i] = theta[i] * 2 * np.pi
        # 自适应调节k
        if y[i] >= 10:
            k[i] = K * np.exp(-2)
        else:
            k[i] = K * np.exp(-y[i] / 5)

    # 位置更新公式，左包围或右包围
    f = []
    l = 0
    yy = copy.deepcopy(y)
    rand = random.random()

    ttheta = copy.deepcopy(theta)
    l = k[0] * rand
    yy[0] = yy[0] * np.exp(-l)
    ttheta[0] = ttheta[0] + l * 2 * np.pi * (-1) ** (random.randint(1, 3))
    f = [yy[0] * np.cos(ttheta[0]), yy[0] * np.sin(ttheta[0])]
    f = np.array(f)

    if n > 2:
        for j in range(n - 2):
            a = (j + 1) % 2
            if a == 1:
                rand = random.random()
                l = k[j + 1] * rand
                yy[j + 1] = yy[j + 1] * np.exp(-l)
                ttheta[j + 1] = ttheta[j + 1] + l * 2 * np.pi * ((-1) ** (random.randint(1, 3)))
                f = np.concatenate((f * abs(np.cos(ttheta[j + 1])), np.array([yy[j + 1] * np.sin(ttheta[j + 1])])))
            elif a == 0:
                rand = random.random()
                l = k[j + 1] * rand
                yy[j + 1] = yy[j + 1] * np.exp(-l)
                ttheta[j + 1] = ttheta[j + 1] + l * 2 * np.pi * ((-1) ** random.randint(1, 3))
                f = np.concatenate((f * abs(np.sin(ttheta[j + 1])), np.array([yy[j + 1] * np.cos(ttheta[j + 1])])))

    f = f.T
    return f

def pdist2(best_x, x):
    best_x = np.array(best_x)
    x = np.array(x)
    a = x - best_x
    b = pow(a, 2)
    c = np.sum(b, axis=1)
    d = pow(c, 0.5)
    return d

def new_min(arr):
    min_data = min(arr)
    key = np.argmin(arr)
    return min_data, key

def new_max(arr):
    max_data = max(arr)
    key = np.argmax(arr)
    return max_data, key

sub = np.array([-50, -50, -50, -50, -50, -50, -50, -50, -50, -50])  # 自变量下限
up = np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50])  # 自变量上限
opt = -1  # -1为最小化，1为最大化
# 程序为最小化寻优，如果是最大化寻优更改排序部分程序即可
n = len(sub)  # 自变量个数
num = 500 * n  # 种群大小
det = 10 * n + 100  # 迭代次数
k = 1.5 + 0.1 * n  # k为最大环绕圈数
R = 0.1 * pow(n, 2)  # 当鲨鱼进入猎物该范围，则直接对猎物位置进行逼近
Mc = (up - sub) * 0.1  # 猎物行动最大范围
x = np.zeros([num, n])
f = np.zeros(num)

for s in range(num):
    rand_data = random.random(n)
    rand_data = np.array(rand_data)
    x[s, :] = sub + (up - sub) * rand_data
    f[s] = fun(x[s, :])

best_x = np.zeros(n)
best_f = 0
# 以最小化为例
if opt == -1:
    best_f, a = new_min(f)  # 记录历史最优值
    best_x = x[a, :]  # 记录历史最优解
elif opt == 1:
    best_f, a = new_max(f)  # 记录历史最优值
    best_x = x[a, :]  # 记录历史最优解

trace = np.array([deepcopy(best_f)])
xx = np.zeros([num, n])
ff = np.zeros(num)

for ii in range(det):
    # 猎物躲避,蒙特卡洛模拟周围1000次，并选择最佳的点作为下一逃跑点
    d = pdist2(best_x, x)
    d.sort()
    z = np.exp(-d[1] / np.mean(Mc))  # 猎物急躁系数
    z = max(z, 0.1)
    best_t = []
    best_c = []
    yx = []
    dx = []
    for i in range(1000):
        m = []
        for iii in range(n):
            randi = random.randint(1, 3)
            a = pow(-1, randi)
            m.append(a)

        m = np.array(m)
        random_rand = random.random(n)
        xd = best_x + Mc * z * ((det - (ii + 1)) / det) * random_rand * m
        xd = np.maximum(sub, xd)
        xd = np.minimum(up, xd)
        if i < 1:
            dx = deepcopy(xd)  # (det-ii)/det表示随着追捕，猎物可逃窜的范围越来越小
        else:
            dx = np.vstack((dx,xd))  # (det-ii)/det表示随着追捕，猎物可逃窜的范围越来越小
        yx=np.hstack((yx,fun(xd)))

    if opt == -1:
        best_t, a = new_min(yx)  # 选择最佳逃跑点
        best_c = dx[a, :]
        if best_t < best_f:
            best_f = best_t
            best_x = best_c
        else:
            pass
    elif opt == 1:
        best_t, a = new_max(yx)  # 选择最佳逃跑点
        best_c = dx[a, :]
        if best_t > best_f:
            best_f = best_t
            best_x = best_c
        else:
            pass

    # 鲸鱼追捕
    for i in range(num):
        # 更新公式
        if np.sqrt(np.sum((x[i, :] - best_x) ** 2)) <= R:
            rand = random.random()
            xx[i, :] = x[i, :] + rand * (x[i, :] - best_x)
            xx[i, :] = np.maximum(sub, xx[i, :])
            xx[i, :] = np.minimum(up, xx[i, :])
            ff[i] = fun(xx[i, :])
        else:
            xx[i, :] = x[i, :] + np.real(jingyu(x[i, :] - best_x, k))
            xx[i, :] = np.maximum(sub, xx[i, :])
            xx[i, :] = np.minimum(up, xx[i, :])
            ff[i] = fun(xx[i, :])

    # 引入上一代进行排序,并重新分配角色
    F = np.hstack((np.array([best_f]), f, ff))
    F= np.array(F)
    X = np.vstack(([best_x], x, xx))
    X=np.array(X)
    temp=np.sort(F,axis=-1,kind='stable')
    if opt == -1:
        F, b = temp, np.argsort(F)  # 按小到大排序
    elif opt == 1:
        F, b = temp[::-1], np.argsort(-F)  # 按大到大排序

    X = X[b, :]
    f = F[:num]
    x = X[:num, :]

    if opt == -1:
        best_f, a = new_min(f)  # 记录历史最优值
    elif opt == 1:
        best_f, a = new_max(f)  # 记录历史最优值

    best_x = x[ a , : ]  # 记录历史最优解
    trace = np.hstack((trace, [best_f]))

print('最优解为：')
print(best_x)
print('最优值为：')
print(float(best_f))
