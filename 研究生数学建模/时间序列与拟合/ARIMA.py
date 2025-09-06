import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
import warnings
from sklearn.metrics import mean_absolute_error,mean_squared_error
warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文

#########################################################################################
df=pd.read_csv('时间序列预测数据集.csv',parse_dates=['Date'])

#将默认索引改为时间索引
data=df.copy()#.iloc[167:,:]
data=data.set_index('Date')

#绘制时序图
plt.plot(data.index,data['Temp'].values)
plt.xticks(rotation=45)
plt.title("时序图")
plt.show()

train=data.iloc[:,:]
test=data.iloc[:,:]  #前闭后开
'''
# 单位根检验-ADF检验
adf_math = sm.tsa.stattools.adfuller(train['Temp'])
print('ADF值:',format(adf_math[0],'.3f'))
print('拒绝程度值:',adf_math[4])
print('ADF值需要小于三个拒绝程度值')

# 白噪声检验
al = acorr_ljungbox(train['Temp'], lags = [6, 12],boxpierce=True)
print('白噪声检验:\n',al)   #LB和BP统计量的P值都小于显著水平（α = 0.05）,所以拒绝序列为纯随机序列的原假设，认为该序列为非白噪声序列
# 计算ACF
acf=plot_acf(train['Temp'])
plt.title("自相关图")
plt.show()

# PACF
pacf=plot_pacf(train['Temp'])
plt.title("偏自相关图")
plt.show()
plt.close()

ACF	 PACF	模型
拖尾	 截尾	 AR
截尾	 拖尾	 MA
拖尾	 拖尾	ARMA
如果说自相关图拖尾，并且偏自相关图在p阶截尾时，此模型应该为AR(p )。
如果说自相关图在q阶截尾并且偏自相关图拖尾时，此模型应该为MA(q)。
如果说自相关图和偏自相关图均显示为拖尾，那么可结合ACF图中最显著的阶数作为q值，选择PACF中最显著的阶数作为p值，最终建立ARMA(p,q)模型。

trend_evaluate = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='n', max_ar=20,max_ma=5)
print('计算中，大概几分钟，别急。\nAIC和BIC前半部分是一样的，BIC考虑了样本数量，样本数量过多时，可有效防止模型精度过高造成的模型复杂度过高。')
print('train AIC(p,q):', trend_evaluate.aic_min_order)
print('train BIC(p,q):', trend_evaluate.bic_min_order)
'''
# 训练
model = sm.tsa.arima.ARIMA(train,order=(8,0,0))  #第一个是p，第三个是q
arima_res=model.fit()
predict=arima_res.predict()
print('MAE:',mean_absolute_error(test['Temp'],predict))
print('mse:',mean_squared_error(test['Temp'],predict))

plt.plot(test.index,test['Temp'])
plt.plot(test.index,predict)
plt.xticks (rotation =45)
plt.legend(['y_true','y_pred'])
plt.show()

# 预测
forecast_res = arima_res.get_forecast(steps=10)      # 预测步数
mean_forecast = forecast_res.predicted_mean          # 预测均值
print("预测均值:", mean_forecast)


# 稳定性分析
def wending():
    noise_level = 0.05  # 设置噪声强度，可以调节大小
    X_train_noisy = train + np.random.normal(loc=0.0, scale=noise_level, size=train.shape)
    model = sm.tsa.arima.ARIMA(X_train_noisy,order=(8,0,0))  #第一个是p，第三个是q
    arima_res=model.fit()
    predict=arima_res.predict()

    plt.plot(test.index,test['Temp'])
    plt.plot(test.index,predict)
    plt.xticks (rotation =45)
    plt.legend(['y_true','y_pred'])
    plt.show()
    print('MAE:',mean_absolute_error(test['Temp'],predict))
    print('mse:',mean_squared_error(test['Temp'],predict))

# 不确定性分析
def buqueding():
    # 残差分析
    residuals = arima_res.resid
    # 对 residuals 随机采样，否则时间太长
    sample_resid = np.random.choice(residuals.ravel(), size=3000, replace=False)
    import seaborn as sns
    sns.histplot(sample_resid, kde=True)
    plt.title("残差正态分布检验")
    plt.show()
    import statsmodels.api as sm
    sm.qqplot(sample_resid, line='45', fit=True)
    plt.title("残差QQ图")  #靠近45度线表明符合正态
    plt.show()
    from scipy.stats import shapiro,normaltest, jarque_bera
    stat, p = normaltest(residuals)
    print('D’Agostino K² test p-value', round(p,5))
    stat, p = shapiro(residuals)
    print('Shapiro-Wilk test p-value:', round(p,5))
    stat, p = jarque_bera(residuals)
    print('Jarque-Bera test p-value:', round(p,5))
    if p > 0.05:
        print("残差近似正态")
    else:
        print("残差偏离正态")
    # 预测区间
    forecast_res = arima_res.get_forecast(steps=10)  # 预测步数
    mean_forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)  # 95% 置信
    #print("预测均值:", mean_forecast)
    #print("置信区间:", conf_int)
    # 绘图
    plt.figure(figsize=(8,5))
    plt.plot(range(len(mean_forecast)), mean_forecast, color="blue", label="预测均值")
    plt.fill_between(range(len(mean_forecast)), conf_int['lower Temp'], conf_int['upper Temp'],
                     color="blue", alpha=0.2, label="95% CI")
    plt.xlabel("X")
    plt.ylabel("预测值")
    plt.title("预测均值与置信区间")
    plt.legend()
    plt.show()

    # 参数化bootstrap
    H = 10        # 预测步数
    B = 50       # bootstrap 次数
    sim_preds = np.zeros((B, H))
    for b in range(B):
        # 使用 simulate 生成整个序列（保持时间依赖）
        sim_series = arima_res.simulate(nsimulations=len(train)+H, anchor=None, initial_state=None)
        # 用原始训练长度拟合模型
        model_b = sm.tsa.arima.ARIMA(sim_series[:len(train)], order=(8,0,0))
        res_b = model_b.fit()
        # 预测未来 H 步
        forecast_b = res_b.get_forecast(steps=H).predicted_mean
        sim_preds[b, :] = forecast_b.values
    # 计算均值和 95% CI
    mean_pred = np.mean(sim_preds, axis=0)
    lower = np.percentile(sim_preds, 2.5, axis=0)
    upper = np.percentile(sim_preds, 97.5, axis=0)
    # 输出
    print("预测均值:", mean_pred)
    print("95% CI 下界:", lower)
    print("95% CI 上界:", upper)
    # 可视化
    plt.figure(figsize=(8,5))
    plt.plot(range(H), mean_pred, color='blue', label='预测均值')
    plt.fill_between(range(H), lower, upper, color='blue', alpha=0.2, label='95% CI')
    plt.xlabel("步数")
    plt.ylabel("预测值")
    plt.title("参数化 Bootstrap 预测均值与 95% CI")
    plt.legend()
    plt.show()

#wending()
buqueding()