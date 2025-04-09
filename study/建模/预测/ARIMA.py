import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
import warnings
from math import log
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文

data = pd.read_csv('时间序列预测数据集.csv',parse_dates=['Date'])
data.set_index('Date',inplace=True)     #将时间设置为行索引

####################################   绘制时序图   ############################################
def picture(data):
    plt.figure(figsize=(16,16))
    for i in range(1,5):
        plt.subplot(4,1,i)
        df = data[f"198{1+2*(i-1)}-01-01":f"198{3+2*(i-1)}-01-01"]
        plt.plot(df.index,df['Temp'].values)
        plt.xticks(rotation=45,size=6)
        if i == 1:
            plt.title("时序图")
    plt.show()
    plt.close()
######################################   时间序列检验      #######################################
def inspect(data):
    # 单位根检验-ADF检验
    ADF = sm.tsa.stattools.adfuller(data['Temp'])
    print('ADF值:', format(ADF[0], '.4f'))
    print('拒绝程度值:', ADF[4])      #ADF值需要小于三个拒绝程度值

    # 白噪声检验
    white_noise = acorr_ljungbox(data['Temp'], lags = [6, 12, 24],boxpierce=True) #lag是需要检验的阶数
    print('白噪声检验:\n',white_noise)   #LB和BP统计量的P值都小于显著水平（α = 0.05）,所以拒绝序列为纯随机序列的原假设，认为该序列为非白噪声序列

    fig = plt.figure(figsize=(16,6))

    ax1 = fig.add_subplot(1,2,1)
    plot_acf(data['Temp'],ax=ax1)
    plt.title("自相关图")        #需要拖尾

    ax2 = fig.add_subplot(1,2,2)
    plot_pacf(data['Temp'],ax=ax2)
    plt.title("偏自相关图")      #需要截尾,确定p

    plt.show()
    plt.close()
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
    X = np.zeros((h,p+1))
    for i in range(h):            #得到X矩阵
        X[i][0] = 1
        for v in range(1,p+1):
            X[i,-v] = df[i+v-1]
    sigma = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y.T)          #最小二乘法估计参数值
    for i in range(h):
        df0[p+i] = sum(sigma * X[i])      #得到所有预测的估计值
    df00 = df.copy()
    df1 = np.zeros(predict_x_n)
    for i in range(predict_x_n):
        df1[i] = sum(sigma[1:] * df00[-p:][::-1])+sigma[0]        #得到所有预测值
        df00 = np.append(df00,df1[i])
    return df0,df1

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
def MA_prediction(data, q, predict_x_n=0):
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
    def MA_AR_prediction(x, p, predict_x_n = 0):           #误差项预测实际就为AR预测，这也是为什么MA模型阶数为0时就转化成AR模型的原因
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
            df0[p+i] = sum(sigma * X[i])        #得到所有预测的估计值
        df00 = df.copy()
        df1 = np.zeros(predict_x_n)
        for i in range(predict_x_n):
            df1[i] = sum(sigma[1:] * df00[-p:][::-1])+sigma[0]       #得到所有预测值
            df00 = np.append(df00,df1[i])
        return df0,df1
    wucha,wucha_predict = MA_AR_prediction(df_wu,q,predict_x_n)
    return wucha,wucha_predict
def sum_s(data,df,q,result_predict):
    data = np.array(data).ravel()
    df = np.array(df).ravel()
    result_predict = np.array(result_predict).ravel()
    if len(data)-len(df)==1+q:
        data0 = np.zeros(len(data))
        data0[0] = data[0]
        for i in range(len(data)-1):
            data0[i+1] = data[i]
        df0 = np.zeros(len(df)+1)
        for i in range(len(df)):
            df0[i+1] = df[i]
        asd = data0[-len(df0):]+df0
        for i in range(len(result_predict)):
            if i == 0:
                result_predict[0]=result_predict[0]+asd[-1]
            else:
                result_predict[i]=result_predict[i]+result_predict[i-1]
                print(result_predict)
        return asd,result_predict
    elif len(data)-len(df)==2+q:
        cha1 = np.diff(data)
        cha1_1,result_predict = sum_s(cha1,df,q,result_predict)
        return sum_s(data,cha1_1,q,result_predict)
    elif len(data)-len(df)==3+q:
        cha1 = np.diff(data)
        cha11 = np.diff(cha1)
        cha1_1,result_predict = sum_s(cha11,df,q,result_predict)
        cha1_1_1,result_predict = sum_s(cha1,cha1_1,q,result_predict)
        return sum_s(data,cha1_1_1,q,result_predict)
def ARIMA(p,q,d,data,predict_n=0):
    df_0 = np.array(data).ravel()
    df = np.diff(df_0,d)
    wucha,wucha_predict = MA_prediction(df, q, predict_n)
    #p_aicc,p_bic = p_finally(10)
    df0,df1 = AR_prediction(df,p,predict_n)
    result = df0[-len(wucha):] - wucha
    result_predict = df1-wucha_predict
    if d != 0:
        result,result_predict = sum_s(df_0,result,q,result_predict)
    return result,result_predict

result,result_predict = ARIMA(4,4,0,data,4)

asd = np.append(result,result_predict)
plt.plot(range(len(data[-60:])),data[-60:],c='b')
plt.plot(range(len(result[-63:])),asd[-63:],c='r')
plt.show()