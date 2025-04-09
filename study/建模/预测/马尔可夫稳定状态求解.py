from scipy.optimize import fsolve
import numpy as np

def func(x):  #方程式联解   #状态转移矩阵,行：a,b,c
    X = np.array([[0.1, 0.2, 0.7],
                          [0.2, 0.3, 0.5],
                          [0.3, 0.4, 0.3]],dtype=float)
    x = np.array(x)
    l = len(x)
    eq = np.array([[-x[i]] for i in range(l)])
    for j in range(l):
        for i in range(l):
            eq[j] += X[i][j]*x[i]
    return eq.ravel()
# 求解方程组
result = fsolve(func, [0.3, 0.4, 0.3])
s = sum(result)
for i in range(len(result)): # 归一化
    result[i] = result[i] / s
print(result)