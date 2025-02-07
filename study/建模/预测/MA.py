import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文

data = pd.read_csv('时间序列预测数据集.csv',parse_dates=['Date'])
data.set_index('Date',inplace=True)     #将时间设置为行索引

def sma(x):
    return np.mean(x)        #简单移动平均

def wma(x):
    x = np.array(x).ravel()
    w = np.arange(1,len(x)+1)      #生成等差数列
    return np.sum(x*w)/(len(x)*(len(x)+1)/2)       #加权移动平均（等距）

def ema(x,i):
    x = np.array(x).ravel()
    if i < 1 and i > 0:
        l = len(x)
        w = np.logspace(l, 0, num=l, base=(1-i))      #生成等比数列
    else:
        print('平滑因子范围错误')
    return np.sum(x*w)/np.sum(w)         #指数移动平均

def ma_prediction(data, q, predict_x_n=0):
##########################   移动平均预测    ################################
    df = np.array(data).ravel()     #将数据转化为numpy格式，并将其转化为一维列表
    df0 = df.copy()         #准备一个副本，存储真实值以及预测值
    for i in range(len(data)-q+1):
        df[q+i-1] = ema(data[i:i+q],0.5)         #获得简单移动平均获得的预测数列

    df_wu = df0 - df        #获得误差项

    df1 = np.zeros(predict_x_n)      #存储预测值
    for i in range(predict_x_n):
        df1[i] = ema(df0[-q:],0.5)        #得到所有预测值
        df0 = np.append(df0,df1[i])
##############################   误差项预测   ###############################
    def AR_prediction(x, p, predict_x_n = 0):           #误差项预测实际就为AR预测，这也是为什么MA模型阶数为0时就转化成AR模型的原因
        df = np.array(x[p:]).ravel()     #将数据转化为numpy格式，并将其转化为一维列表
        df0 = df.copy()         #准备一个副本，为了存储预测的数据
        Y = df[p:].copy()
        h = len(df0)-p
        X = np.zeros((h,p+1))
        for i in range(h):            #得到X矩阵
            X[i][0] = 1
            for v in range(1,p+1):
                X[i,-v] = df[i+v-1]
        sigma = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y.T)          #最小二乘法估计参数值
        for i in range(h):
            df0[p+i] = sum(np.multiply(sigma , X[i]))        #得到所有预测的估计值
        df00 = df.copy()
        df1 = np.zeros(predict_x_n)
        for i in range(predict_x_n):
            df1[i] = sum(np.multiply(sigma , df00[-p:][::-1]))        #得到所有预测值
            df00 = np.append(df00,df1[i])
        return df0,df1
    wucha,wucha_predict = AR_prediction(df_wu,q,predict_x_n)
    result = df[-len(wucha):] - wucha
    predict = df1 - wucha_predict
    return result,predict

result,predict = ma_prediction(data, 6, 0)

plt.figure()
plt.plot(range(60),data[-60:],c = 'black')
plt.plot(range(60),result[-60:],c='red')
plt.show()