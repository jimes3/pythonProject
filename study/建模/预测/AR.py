from math import log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文

data = pd.read_csv('时间序列预测数据集.csv',parse_dates=['Date'])
data.set_index('Date',inplace=True)     #将时间设置为行索引
#data = data["1981-01-01":"1982-01-01"]
####################################   绘制时序图   ############################################
def picture(data):
    plt.figure(figsize=(16,16))
    for i in range(1,5):
        plt.subplot(4,1,i)
        df = data[f"198{1+2*(i-1)}-01-01":f"198{3+2*(i-1)}-01-01"]
        plt.plot(df.index,df['Temp'])
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
    plt.title("自相关图")        #需要拖尾,确定q

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

####################################     模型预测     ##########################################
def AR_prediction(x, p, predict_x_n = 0):
    df = np.array(x).ravel()     #将数据转化为numpy格式，并将其转化为一维列表
    df0 = df.copy()         #准备一个副本，为了存储预测的数据
    Y = df[p:].copy()
    h = len(df0)-p
    X = np.zeros((h,p+1))
    for i in range(h):            #得到X矩阵
        X[i][0] = 1
        for v in range(1,p+1):
            X[i][-v] = df[i+v-1]
    sigma = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y.T)          #最小二乘法估计参数值
    print(sigma)
    for i in range(h):
        df0[p+i] = sum(sigma * X[i])        #得到所有预测的估计值
    df00 = df.copy()
    df1 = np.zeros(predict_x_n)
    for i in range(predict_x_n):
        df1[i] = sum(np.multiply(sigma , df00[-p:][::-1]))        #得到所有预测值
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

#################################   得到最终结果   ###################################
#images(data)
#inspect(data)

#p_aicc,p_bic = p_finally(30)
#print(p_aicc)
df0,df1 = AR_prediction(data,4,0)

plt.figure()
plt.plot(range(50),data[-50:],c = 'black')
plt.plot(range(50),df0[-50:],c='red')
plt.show()
