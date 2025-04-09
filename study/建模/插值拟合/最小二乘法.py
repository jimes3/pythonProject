import numpy as np


def lsm(X,Y):
    X = np.array(X)
    Y = np.array(Y).ravel()
    sigma = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y.T)          #最小二乘法估计参数值
    df = sigma*X
    return sigma,df

x = [[1],[2],[3],[4],[5],[6]]     #注意这里必须是二维数据
y = [1,2,3,4,5,6]
sigma,df = lsm(x,y)
print(sigma,df)