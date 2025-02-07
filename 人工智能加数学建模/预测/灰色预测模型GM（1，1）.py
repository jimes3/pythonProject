'''灰色预测模型GM(1,1)是在数学建模比赛中常用的预测值方法，常用于中短期符合指数规律的预测。
    一阶微分方程，且只含有一个变量的灰色模型'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

# 灰色预测模型
def greyModel(dataVec, predictLen):
    "Grey Model for exponential prediction"
    # dataVec = [1, 2, 3, 4, 5, 6]
    # predictLen = 5
    import numpy as np
    from scipy import io, integrate, linalg, signal
    from scipy.sparse.linalg import eigs
    from scipy.integrate import odeint

    x0 = np.array(dataVec, float)
    n = x0.shape[0]
    x1 = np.cumsum(x0)
    B = np.array([-0.5 * (x1[0:-1] + x1[1:]), np.ones(n-1)]).T
    Y = x0[1:]
    u = linalg.lstsq(B, Y)[0]

    def diffEqu(y, t, a, b):
        return np.array(-a * y + b)

    t = np.arange(n + predictLen)
    sol = odeint(diffEqu, x0[0], t, args=(u[0], u[1]))
    sol = sol.squeeze()
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
x_hat = greyModel(x+c, 5)-c    #5为预测的数量

print('未来5次的数据预测为：',x_hat[-5:])
# 画图
t1 = range(x.size)
t2 = range(x_hat.size)
plt.plot(t1, x, color='r', linestyle="-", marker='*', label='True')
plt.plot(t2, x_hat, color='b', linestyle="--", marker='.', label="Predict")
plt.legend(loc='upper right')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.title('Prediction by Grey Model (GM(1,1))')

res = regressionAnalysis(x_hat[0:x.size], x)
print('回归系数a，b分别为：',res[0]) # 回归系数a, b
print('残差平方和为：',res[1]) # residuals 残差平方和
plt.show()
'''

#---------------- 输入数据 -------------------
df = pd.read_csv('zong.csv')
df = df.iloc[:,1]
df = np.array(df)
df.reshape(-1)
print(df)
xxx = np.array([
    [83.0, 79.8, 78.1, 85.1, 86.6, 88.2, 90.3, 86.7, 93.3, 92.5, 90.9, 96.9],
    [101.7, 85.1, 87.8, 91.6, 93.4, 94.5, 97.4, 99.5, 104.2, 102.3, 101.0, 123.5],
    [92.2, 114.0, 93.3, 101.0, 103.5, 105.2, 109.5, 109.2, 109.6, 111.2, 121.7, 131.3],
    [105.0, 125.7, 106.6, 116.0, 117.6, 118.0, 121.7, 118.7, 120.2, 127.8, 121.8, 121.9],
    [139.3, 129.5, 122.5, 124.5, 135.7, 130.8, 138.7, 133.7, 136.8, 138.9, 129.6, 133.7],
    [137.5, 135.3, 133.0, 133.4, 142.8, 141.6, 142.9, 147.3, 159.6, 162.1, 153.5, 155.9]])
xx = []
for i in range(6):
    s = np.mean(xxx[i])
    xx.append(s)#得到前六年的月平均值

x = np.array(xx)
c = greyModelPreprocess(x)
x_hat = greyModel(x+c, 1)-c#预测第七年的月平均值

sumx = 12 * x_hat[-1]

sumxx = 0
for i in range(6):
    sumxx = sumxx + sum(xxx[i])
bi = []
for i in range(12):
    ss = sum(xxx)[i] / sumxx
    bi.append(ss)

predict_new = np.multiply(bi,sumx)
print(predict_new)

no_predict = [163.2,159.7,158.4,145.2,124,144.1,157,162.6,171.8,180.7,173.5,176.5]
t1 = range(12)
t2 = range(12)
plt.plot(t1, no_predict, color='r', linestyle="-", marker='*', label='True')
plt.plot(t2, predict_new, color='b', linestyle="--", marker='.', label="Predict")
plt.legend(loc='upper right')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.title('Prediction by Grey Model (GM(1,1))')
plt.show()
'''