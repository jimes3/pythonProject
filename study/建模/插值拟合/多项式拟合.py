import numpy as np
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出


def pf(x,y,n=0):          #n等于0就是最小二乘估计，不等于0为岭回归
    x = np.array(x)
    if n != 0:
        x_x = np.ones((len(x),n+1))
        for i in range(len(x)):
            for v in range(1,n+1):
                x_x[i][v] = x[i]**v
    else:
        x_x = x
    y = np.array(y).ravel()
    a,b = x_x.shape
    I = np.identity(b)
    sigma = np.linalg.inv(x_x.T.dot(x_x) + n * I).dot(x_x.T).dot(y.T)          #最小二乘法估计参数值
    df = sigma*x
    df = np.sum(df,axis=1)
    return sigma,df

x = [[1],[2],[3],[4]]     #注意这里必须是二维数据
y = [0.76,0.35,0.31,0.17]
sigma,df = pf(x,y,0)
print(sigma,'\n',df)