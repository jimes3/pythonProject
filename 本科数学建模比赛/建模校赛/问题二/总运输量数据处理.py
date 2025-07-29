import ydata_profiling as pp
import webbrowser
import warnings
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer,KNNImputer
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pylab
from pmdarima.arima import auto_arima
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文
##########################     导入数据      #################################
df = pd.read_excel('附件1(Attachment 1)2023-51MCM-Problem B.xlsx',header = 0)
df = df.set_index("日期(年/月/日) (Date Y/M/D)")
df = df.resample("B").sum()
df000 = df.copy()

# 通过Z-Score方法判断异常值
df_zscore = df.copy()  # 复制一个用来存储Z-score得分的数据框
cols = df.columns  # 获得数据框的列名
for col in cols:  # 循环读取每列
    df_col = df[col]  # 得到每列的值
    z_score = (df_col - df_col.mean()) / df_col.std()  # 计算每列的Z-score得分
    df_zscore[col] = z_score.abs() > 2.2  # 判断Z-score得分是否大于2.2，如果是则是True，否则为False
# print('是否为异常值')
# print (df_zscore)
columns = list(df)
for col in columns:
    index = df_zscore[df_zscore[col] == True].index.tolist()
    for i in index:
        df.loc[i, col] = np.nan
# 使用KNN算法填充缺失值
imp = KNNImputer(n_neighbors=10)
# n_neighbors的值越大，模型考虑的邻居数量也就越多，预测结果也会更加准确，但同时也会使模型计算复杂度更高。
df = imp.fit_transform(df)
df = pd.DataFrame(df)

for i in range(len(df)):
    df000.iloc[i,-1]=df.iloc[i]

# 定义要删除的时间范围
start_date = '2018-11-01'
end_date = '2018-12-31'
# 使用 loc[] 方法选择要删除的行
rows_to_drop = df000.loc[start_date:end_date]
# 使用 drop() 方法删除选定的行
df1 = df000.drop(rows_to_drop.index)
# 定义要删除的时间范围
start_date1 = '2018-06-01'
end_date1 = '2018-06-30'
# 使用 loc[] 方法选择要删除的行
rows_to_drop1 = df000.loc[start_date1:end_date1]
# 使用 drop() 方法删除选定的行
data = df1.drop(rows_to_drop1.index)
data = np.array(data).ravel()
#print(data)
#data.to_csv('zong.csv', sep=',',header=False)
# 创建一个时间索引
date_range = pd.date_range(start='2018-07-19', periods=len(data), freq='D')
# 将一维数组转换为 Series 对象，并设置索引
s = pd.Series(data, index=date_range)
ddd = np.array(data)
#绘制时序图
plt.plot(s.index,ddd)
plt.xticks(rotation=45)
plt.title("总运输数量")
plt.show()

train = ddd
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


# 进行ARIMA模型自动调参
model = auto_arima(ddd, seasonal=False, trace=False)
# 输出最优模型参数
print('最优参数为：\n',model.order)
a = model.order[0]
b = model.order[1]
c = model.order[2]
model = sm.tsa.arima.ARIMA(train,order=(a,b,c))  #第一个是p，第三个是q
arima_res=model.fit()
arima_res.summary()
predict=arima_res.predict(len(ddd),len(ddd)+1)   #前闭后闭
print(predict)
plt.show()



