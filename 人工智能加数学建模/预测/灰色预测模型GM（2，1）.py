'''根据资料，GM(2,1)适用于非单调的摆动发展序列，或有饱和的S型序列。
    但是从图像上观察，数据预测由于为指数类型，变化过于夸张、预测趋势也有偏离的状况，可能实用性和普适性并不如GM(1,1)
    二阶微分方程，且只含有一个变量的灰色模型'''
import numpy as np
import matplotlib.pyplot as plt

# 线性平移预处理，确保数据级比在可容覆盖范围
def greyModelPreprocess(dataVec):
    "Set linear-bias c for dataVec"
    import numpy as np
    from scipy import io, integrate, linalg, signal
    from scipy.sparse.linalg import eigs
    from scipy.integrate import odeint

    c = 0
    x0 = np.array(dataVec, float)
    n = x0.shape[0]
    L = np.exp(-2/(n+1))
    R = np.exp(2/(n+2))
    xmax = x0.max()
    xmin = x0.min()
    if (xmin < 1):
        x0 += (1-xmin)
        c += (1-xmin)
    xmax = x0.max()
    xmin = x0.min()
    lambda_ = x0[0:-1] / x0[1:]  # 计算级比
    lambda_max = lambda_.max()
    lambda_min = lambda_.min()
    while (lambda_max > R or lambda_min < L):
        x0 += xmin
        c += xmin
        xmax = x0.max()
        xmin = x0.min()
        lambda_ = x0[0:-1] / x0[1:]
        lambda_max = lambda_.max()
        lambda_min = lambda_.min()
    return c

# 灰色预测模型GM(2,1)
def greyModel2(dataVec, predictLen):
    "Grey Model for exponential prediction"
    #dataVec = [1, 2, 3, 4, 5, 6]
    #predictLen = 5
    import numpy as np
    import sympy as sy
    from scipy import io, integrate, linalg, signal

    x0 = np.array(dataVec, float)
    n = x0.shape[0]
    a_x0 = np.diff(x0) # 1次差分序列
    x1 = np.cumsum(x0) # 1次累加序列
    z = 0.5 * (x1[0:-1] + x1[1:]) # 均值生成序列
    B = np.array([-x0[1:], -z, np.ones(n-1)]).T
    u = linalg.lstsq(B, a_x0)[0]

    def diffEqu(x, f, a1, a2, b):
        return sy.diff(f(x), x, 2) + a1*sy.diff(f(x), x) + a2*f(x) - b # f''(x)+a1*f'(x)+a2*f(x)=b 二阶常系数齐次微分方程

    t = np.arange(n + predictLen)
    x = sy.symbols('x')  # 约定变量
    f = sy.Function('f')  # 约定函数
    eq = sy.dsolve(diffEqu(x, f, u[0], u[1], u[2]), f(x), ics={f(t[0]): x1[0], f(t[n-1]): x1[-1]})
    f = sy.lambdify(x, eq.args[1], 'numpy')
    sol = f(t)
    res = np.hstack((x0[0], np.diff(sol)))
    return res

def regressionAnalysis(xVec, yVec):
    import numpy as np
    from scipy import linalg

    x = np.array([xVec, np.ones(xVec.size)]).T
    u = linalg.lstsq(x, yVec)
    return u
#-------------- 输入数据 -------------------
x = np.array([-18, 0.34, 4.68, 8.49, 29.84, 50.21, 77.65, 109.36])
c = greyModelPreprocess(x)
x_hat = greyModel2(x+c, 5)-c    #5为预测的数量

print('未来5次的数据预测为：',x_hat[-5:])
# 画图
t1 = range(x.size)
t2 = range(x_hat.size)
plt.plot(t1, x, color='r', linestyle="-", marker='*', label='True')
plt.plot(t2, x_hat, color='b', linestyle="--", marker='.', label="Predict")
plt.legend(loc='upper right')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.title('Prediction by Grey Model (GM(2,1))')

res = regressionAnalysis(x_hat[0:x.size], x)
print('回归系数a，b分别为：',res[0]) # 回归系数a, b
print('残差平方和为：',res[1]) # residuals 残差平方和
plt.show()