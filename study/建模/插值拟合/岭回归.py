import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文



def rr(X,Y,lamla=0):       #当lamla为0时，就变成了最小二乘回归
    X = np.array(X)
    Y = np.array(Y).ravel()
    a,b = X.shape
    I = np.identity(b)
    sigma = np.linalg.inv(X.T.dot(X)+lamla*I).dot(X.T).dot(Y.T)          #估计参数值
    df = sigma*x
    return sigma,df

x = [[1],[2],[3],[4],[5],[6]]     #注意这里必须是二维数据
y = [1,2,3,4,5,6]
n = 10
mse = []
for k in range(1,n):
    sigma,df = rr(x,y,k)
    mse.append(mean_squared_error(y,df))
plt.figure()
plt.plot(range(1,n),mse,c='red')          #观察岭迹图确定岭系数
# 用岭迹法：
#（1）各回归系数的岭估计基本稳定；
#（2）用最小二乘估计时符号不合理的回归系数，其岭估计的符号变得合理；
#（3）回归系数没有不合乎经济意义的绝对值；
#（4）残差平方和增大不太多。
plt.show()