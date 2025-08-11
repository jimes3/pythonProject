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
df=pd.read_csv('zong.csv',parse_dates=['DATA_TIME'])
df.info()

#将默认索引改为时间索引
data=df.copy()#.iloc[167:,:]
data=data.set_index('DATA_TIME')

#绘制时序图
plt.plot(data.index,data['POWER'].values)
plt.xticks(rotation=45)
plt.title("时序图")
#plt.show()

train=data.iloc[:274,:]
test=data.iloc[274:,:]  #前闭后开

# 单位根检验-ADF检验
adf_math = sm.tsa.stattools.adfuller(train['POWER'])
print('ADF值:',format(adf_math[0],'.3f'))
print('拒绝程度值:',adf_math[4])
print('ADF值需要小于三个拒绝程度值')

# 白噪声检验
al = acorr_ljungbox(train['POWER'], lags = [6, 12],boxpierce=True)
print('白噪声检验:\n',al)   #LB和BP统计量的P值都小于显著水平（α = 0.05）,所以拒绝序列为纯随机序列的原假设，认为该序列为非白噪声序列
# 计算ACF
acf=plot_acf(train['POWER'])
plt.title("自相关图")
#plt.show()

# PACF
pacf=plot_pacf(train['POWER'])
plt.title("偏自相关图")
plt.show()
plt.close()
'''
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
model = sm.tsa.arima.ARIMA(train,order=(8,0,0))  #第一个是p，第三个是q
arima_res=model.fit()
arima_res.summary()

predict=arima_res.predict(274,276)
print(predict)#前闭后闭
plt.plot(test.index,test['POWER'])
plt.plot(test.index,predict)
plt.xticks (rotation =45)
plt.legend(['y_true','y_pred'])
plt.show()
print(len(predict))
print(predict)

print('MAE:',mean_absolute_error(test['POWER'],predict))
print('mse:',mean_squared_error(test['POWER'],predict))
