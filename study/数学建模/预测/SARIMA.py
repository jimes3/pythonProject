import copy
from math import log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文


#####################################    信息准则    ###########################################
def information(df,df0,p):
    n = len(df)
    mse = mean_squared_error(df,df0)
    #aic = n * log(mse) + 2 * p
    aicc = log(mse) + (n+p)/(n-p-2)
    bic = n * log(mse) + p * log(n)
    return aicc,bic

def AR_prediction(x, p, predict_x_n = 0):
    df = np.array(x).ravel()     #将数据转化为numpy格式，并将其转化为一维列表
    df0 = df.copy()         #准备一个副本，为了存储预测的数据
    Y = df[p:].copy()
    h = len(df0)-p
    X = np.zeros((h,p))
    for i in range(h):            #得到X矩阵
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
    return df0,df1,sigma
def SAR(x,p,s,predict_x_n=0):
    x = np.array(x).ravel()
    df0 = x.copy()
    Y = x[s*p:].copy()
    h = len(x) - p*s
    X = np.zeros((h,p))
    for t in range(h):
        for i in range(1,p+1):
            X[t][-i] = x[p*s+t-i*s]
    sigma = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y.T)          #最小二乘法估计参数值
    for i in range(h):
        df0[p*s+i] = sum(sigma * X[i])        #得到所有预测的估计值
    df00 = x.copy()
    df1 = np.zeros(predict_x_n)
    for i in range(predict_x_n):
        df1[i] = sum(np.multiply(sigma , df00[-p:][::-1]))        #得到所有预测值
        df00 = np.append(df00,df1[i])
    return df0,df1,sigma

#########################################   阶数确定   ##############################################
def p_finally(n):
    jieguo = []
    for i in range(1,n):
        df0,df1 = AR_prediction(data,i)
        aicc,bic = information(data,df0,i)
        jieguo.append([i,aicc,bic])
    jieguo_aicc = sorted(jieguo,reverse=False, key=lambda x: x[1])  #以aicc排序
    jieguo_bic = sorted(jieguo,reverse=False, key=lambda x: x[2])  #以bic排序
    return jieguo_aicc[0][0],jieguo_bic[0][0]

def MA_prediction(data, q, predict_x_n=0):
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
    ##########################   移动平均预测    ################################
    df = np.array(data).ravel()     #将数据转化为numpy格式，并将其转化为一维列表
    df0 = df.copy()         #准备一个副本，存储真实值以及预测值
    for i in range(len(df0)-q+1):
        df[q+i-1] = ema(df0[i:i+q],0.5)         #获得简单移动平均获得的预测数列

    df_wu = df0 - df        #获得误差项

    df1 = np.zeros(predict_x_n)      #存储预测值
    for i in range(predict_x_n):
        df1[i] = ema(df0[-q:],0.5)        #得到所有预测值
        df0 = np.append(df0,df1[i])
    ##############################   误差项预测   ###############################
    def MA_AR_prediction(x, p, predict_x_n = 0):           #误差项预测实际就为AR预测，这也是为什么MA模型阶数为0时就转化成AR模型的原因
        df = np.array(x[p:]).ravel()     #将数据转化为numpy格式，并将其转化为一维列表
        df0 = df.copy()         #准备一个副本，为了存储预测的数据
        Y = df[p:].copy()
        h = len(df0)-p
        X = np.zeros((h,p))
        for i in range(h):            #得到X矩阵
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
        return df0,df1,sigma
    wucha,wucha_predict,sigma = MA_AR_prediction(df_wu,q,predict_x_n)
    return wucha,wucha_predict,sigma,df_wu

def SMA(data,q,s,predict_x_n=0):
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
    ##########################   移动平均预测    ################################
    df = np.array(data).ravel()     #将数据转化为numpy格式，并将其转化为一维列表
    df0 = df.copy()         #准备一个副本，存储真实值以及预测值
    h = len(data) - q*s
    X = np.zeros((h,q))
    for t in range(h):
        for i in range(1,q+1):
            X[t][-i] = df[q*s+t-i*s]
    for i in range(h):
        df[q*s+i] = ema(X[i],0.5)         #获得简单移动平均获得的预测数列

    df_wu = df0 - df        #获得误差项

    df1 = np.zeros(predict_x_n)      #存储预测值
    for i in range(predict_x_n):
        df1[i] = ema(df0[-q:],0.5)        #得到所有预测值
        df0 = np.append(df0,df1[i])
    ##############################   误差项预测   ###############################
    def SAR(x,p,s,predict_x_n=0):
        x = np.array(x[p*s:]).ravel()
        df0 = x.copy()
        Y = x[s*p:].copy()
        h = len(x) - p*s
        X = np.zeros((h,p))
        for t in range(h):
            for i in range(1,p+1):
                X[t][-i] = x[p*s+t-i*s]
        sigma = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y.T)          #最小二乘法估计参数值
        for i in range(h):
            df0[p*s+i] = sum(np.multiply(sigma , X[i]))        #得到所有预测的估计值
        df00 = x.copy()
        df1 = np.zeros(predict_x_n)
        for i in range(predict_x_n):
            df1[i] = sum(np.multiply(sigma , df00[-p:][::-1]))        #得到所有预测值
            df00 = np.append(df00,df1[i])
        return df0,df1,sigma
    wucha,wucha_predict,sigma = SAR(df_wu,q,s,predict_x_n)
    return wucha,wucha_predict,sigma
def SARIMA(p,q,P,Q,data,m):
    #p_aicc,p_bic = p_finally(10)
    df0,df1,sigma = AR_prediction(data,p,0)
    wucha,wucha_predict,sigma1,df_wu = MA_prediction(data, q, 0)
    a,b,sigma2 = SAR(data,P,m,0)
    c,d,sigma3 = SMA(data,Q,m,0)
    sigma = sigma[::-1]
    sigma1 = sigma1[::-1]
    sigma2 = sigma2[::-1]
    sigma3 = sigma3[::-1]
    sigma22 = np.append(np.array([1]),sigma2)
    sigma33 = np.append(np.array([1]),sigma3)
    result = []
    for t in range(30,len(data)+1):
        sar = 0
        for p in range(p):
            for P in range(P+1):
                sar += sigma[p] * data[t-p-1-P*m] * sigma22[P]
        sar += sum(sigma2 * np.array([data[t-i*m] for i in range(1,P+1)]))
        sma = 0
        for q in range(q):
            for Q in range(Q+1):
                sma += sigma1[q] * df_wu[t-q-1-Q*m] * sigma33[Q]
        sma += sum(sigma3 * np.array([df_wu[t-i*m] for i in range(1,Q+1)]))
        result.append(sar + sma)
    return result


data0 = pd.read_csv('时间序列预测数据集.csv',parse_dates=['Date'])
data0.set_index('Date',inplace=True)     #将时间设置为行索引
data=np.array(copy.deepcopy(data0)).ravel()
#data = data.reshape((10,365))    #季节性差分
#data=np.diff(data,1)
#data = data.ravel()
result = SARIMA(6,7,2,2,data,365)
plt.figure(figsize=(15, 7.5))
plt.plot(range(500),data[-500:],c = 'black',label='actual')
plt.plot(range(500),result[-500:],c='red',label='model')
plt.show()

print('mse:',mean_squared_error(result,data[-len(result):]))
