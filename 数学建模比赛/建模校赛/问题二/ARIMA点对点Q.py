import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
import warnings
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
#########################################################################################
df = pd.read_excel('附件1(Attachment 1)2023-51MCM-Problem B.xlsx',header = 0)
index = df[(df['发货城市 (Delivering city)'] == 'Q') & (df['收货城市 (Receiving city)'] == 'V')].index.tolist()
#将默认索引改为时间索引
data=df.iloc[index,:].copy()
data=data.set_index('日期(年/月/日) (Date Y/M/D)')
data = data.iloc[:,-1]
pdates = pd.date_range(start="2018-04-19", end="2019-04-17")
data = data.reindex(pdates)
#使用KNN算法填充缺失值
imp=KNNImputer(n_neighbors=10)
#n_neighbors的值越大，模型考虑的邻居数量也就越多，预测结果也会更加准确，但同时也会使模型计算复杂度更高。
df=imp.fit_transform(np.array(data).reshape(-1,1))
for i in range(len(df)):
    data.iloc[i]=df[i]
# 定义要删除的时间范围
start_date = '2018-11-01'
end_date = '2018-12-31'
# 使用 loc[] 方法选择要删除的行
rows_to_drop = data.loc[start_date:end_date]
# 使用 drop() 方法删除选定的行
data = data.drop(rows_to_drop.index)
# 定义要删除的时间范围
start_date1 = '2018-06-01'
end_date1 = '2018-06-30'
# 使用 loc[] 方法选择要删除的行
rows_to_drop = data.loc[start_date1:end_date1]
# 使用 drop() 方法删除选定的行
data = data.drop(rows_to_drop.index)
pdates = pd.date_range(start="2018-07-19", end="2019-04-17")
df = data.reindex(pdates)
for i in range(len(df)):
    df.iloc[i]=data[i]
df.to_csv('df.csv')
print(df)
ddd = np.array(df)
#绘制时序图
plt.plot(df.index,ddd)
plt.xticks(rotation=45)
plt.title("时序图")
plt.show()

train=ddd
# 单位根检验-ADF检验
adf_math = sm.tsa.stattools.adfuller(train)
print('ADF值:',format(adf_math[0],'.3f'))
print('拒绝程度值:',adf_math[4])
print('ADF值需要小于三个拒绝程度值')

# 白噪声检验
al = acorr_ljungbox(train, lags = [6, 12],boxpierce=True)
print('白噪声检验:\n',al)   #LB和BP统计量的P值都小于显著水平（α = 0.05）,所以拒绝序列为纯随机序列的原假设，认为该序列为非白噪声序列
# 计算ACF
acf=plot_acf(train)
plt.title("自相关图")

# PACF
pacf=plot_pacf(train)
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
'''
w = 1
if w==0:
    print('计算中，大概几分钟，别急。\nAIC和BIC前半部分是一样的，BIC考虑了样本数量，样本数量过多时，可有效防止模型精度过高造成的模型复杂度过高。')
    trend_evaluate = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='n', max_ar=20,max_ma=5)
    print('train AIC(p,q):', trend_evaluate.aic_min_order)
    print('train BIC(p,q):', trend_evaluate.bic_min_order)

model = sm.tsa.arima.ARIMA(train,order=(1,0,0))  #第一个是p，第三个是q
arima_res=model.fit()
arima_res.summary()
predict=arima_res.predict(273)   #前闭后闭
print(predict)
