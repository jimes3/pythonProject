import numpy as np
import math
import matplotlib.pyplot  as plt
from copy import deepcopy
import plotly.graph_objects as go
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出


X = np.array([-18, 0.34, 4.68, 8.49, 29.84, 50.21, 77.65, 109.36])
x = np.array([[83.0, 79.8, 78.1, 85.1, 86.6, 88.2, 90.3, 86.7, 93.3, 92.5, 90.9, 96.9],
              [101.7, 85.1, 87.8, 91.6, 93.4, 94.5, 97.4, 99.5, 104.2, 102.3, 101.0, 123.5],
              [92.2, 114.0, 93.3, 101.0, 103.5, 105.2, 109.5, 109.2, 109.6, 111.2, 121.7, 131.3],
              [105.0, 125.7, 106.6, 116.0, 117.6, 118.0, 121.7, 118.7, 120.2, 127.8, 121.8, 121.9],
              [139.3, 129.5, 122.5, 124.5, 135.7, 130.8, 138.7, 133.7, 136.8, 138.9, 129.6, 133.7],
              [137.5, 135.3, 133.0, 133.4, 142.8, 141.6, 142.9, 147.3, 159.6, 162.1, 153.5, 155.9]])
def grey_model(X,form,y):      #X为代预测的数列，form为预测的数量,y为是否绘图，0不画，1画
    x_0 = deepcopy(X)
    ##########################################          级比检验         ###########################################
    x_n = [X[i] / X[i + 1] for i in range(len(X) - 1)]       #计算原数列的级比
    if any(n <= math.exp(-2 / (len(X) + 1)) for n in x_n)==True or any(n >= math.exp(-2 / (len(X) + 2)) for n in x_n)==True:
        print('______未通过级比检验______')
        ####################     级比检验不通过处理      ####################
        i = 0
        while True:
            X += 10      #以10为步长慢慢给原数列加数值
            x_n_new = [X[i] / X[i + 1] for i in range(len(X) - 1)]     #计算每一个新的数列的级比
            if any(n <= np.exp(-2 / (len(X) + 1)) for n in x_n_new)==True or any(n >= np.exp(2 / (len(X) + 1)) for n in x_n_new)==True:
                i += 1
                continue
            else:
                print('修正数列为：\n',X)
                sc = 10*(i+1)
                print('修正值为：\n',sc)
                break
    else:
        print("_______通过级比检验______")

    ######################################     模型构建与求解     #######################################
    X_sum = X.cumsum()   # 重新求得累加数列

    z_n = [(X_sum[i] + X_sum[i + 1]) / 2 for i in range(len(X_sum)-1)]   # 紧邻均值序列

    ############# 最小二乘法计算 ###############
    Y = [X[i] for i in range(1,len(X))]  #生成数组Y
    Y = np.array(Y)   #将数组转化为numpy数组
    Y = Y.reshape(-1,1)   #转换格式

    B = [-z_n[i] for i in range(len(z_n))]  #生成数组B
    B = np.array(B)   #将数组转化为numpy数组
    B = B.reshape(-1,1)   #转换格式
    c = np.ones((len(B),1))    #生成数值全为1的一列数组
    B = np.hstack((B,c))     #将两者相连

    parameters = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)  # 通过numpy求出参数（最小二乘法）
    a = parameters[0,0]   #索引到a参数
    b = parameters[1,0]   #索引到b参数
    print("a=",a)   # 打印结果
    print("b=",b)

    #生成预测模型#####################################################        要改在这改     ################################
    b_a = b/a  #提前计算好常数项，减少后续程序的计算
    model_predict = [(X[0] - b_a) * np.exp(-a * k) + b_a for k in range(len(X)+form)]  #生成预测数列

    list_predict = np.concatenate(([model_predict[0]], np.diff(model_predict))) - sc # 先累减获得预测数列，再做差得到真的预测序列

    print("预测数列为:\n",list_predict)
    list_predict = [float(format(i,'.4f')) for i in list_predict]
    if y == 1:
        #####################################     绘图     ####################################
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(X))),y=x_0,         #添加x,y
                                 mode="markers+lines+text",text=x_0,textposition="top center",  #设置坐标点显示
                                 name='原数列'))  #设置标签名
        fig.add_trace(go.Scatter(x=list(range(len(X)+form)),y=list_predict,        #添加x,y
                                 mode="markers+lines+text",text=list_predict,textposition="top center",  #设置坐标点显示
                                 name='预测数列'))   #设置标签名
        fig.update_layout(title="灰色预测",     #设置图表名
                          xaxis=dict(title="时间序号"),    #设置x轴参数
                          yaxis=dict(title="值"),      #设置y轴参数
                          title_x=0.5,  # 将标题居中
                          title_y=0.94,  # 将标题置于上方
                          template="plotly")
        fig.show()

    ########################################      模型检验      #######################################
    G = np.array(list_predict[:len(list_predict)-form])  #生成与真实数列对应的预测数列
    e = x_0 - G  #获得每一个预测值的残差
    q = abs(e / x_0)  # 获得每一个预测值的相对误差
    print("相对误差数列为:\n",q)
    q_q = q.sum()/(len(q)-1)   #求得平均相对误差
    print(f"平均相对误差为:\n{q_q:.4f}")

    S0 = np.sqrt(np.var(x_0))   #原数列的标准差
    S1 = np.sqrt(np.var(e))    #残差数列的标准差
    print('后验差比值C为:\n',S1 / S0)

    E = e.sum()/(len(e)-1)    #计算残差的均值
    yu_zhi = 0.6745*S0
    g = 0
    for i in range(len(e)):
        if abs(e[i]-E) < yu_zhi:
            g += 1
    p = g/len(e)
    print('小概率误差为:\n',p)

    list_p = list_predict[-form:]       #获得我们预测的值
    return list_p

def grey_models(x):
    av = np.array([np.mean(x[i]) for i in range(len(x))])    #求得每一个过程的平均值
    av_predict = grey_model(av,1,0)          #求得预测过程的平均值
    sum_predict = len(x[0]) * av_predict[0]       #计算获得预测过程的总和
    x_x = x.T   #将数组转置

    pre = np.array([grey_model(x_x[i],1,0) for i in range(len(x_x))])       #获得每个过程同一时间在预测过程中的预测值
    pre = pre.ravel()

    c_predict = np.array(pre/sum(pre))     #获得预测过程每一个的预测占比
    c_predict = c_predict.ravel()
    predict = c_predict * sum_predict
    return predict

list_p = grey_model(X,3,1)
print(list_p)
#predict = grey_models(x)
#print(predict)