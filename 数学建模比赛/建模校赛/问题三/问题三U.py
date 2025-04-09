import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
import warnings
from pmdarima.arima import auto_arima
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
df = pd.read_excel('附件2(Attachment 2)2023-51MCM-Problem B.xlsx',header = 0)
index = df[(df['发货城市 (Delivering city)'] == 'U') & (df['收货城市 (Receiving city)'] == 'O')].index.tolist()
#将默认索引改为时间索引
data=df.iloc[index,:].copy()
data=data.set_index('日期(年/月/日) (Date Y/M/D)')
data = data.iloc[:,-1]
me = data.mean()
#print(me)
pdates = pd.date_range(start="2020-04-28", end="2023-04-27")
data = data.reindex(pdates)
data.replace(np.nan,-me,inplace=True)

# 定义要删除的时间范围
start_date = '2020-11-01'
end_date = '2020-12-31'
# 使用 loc[] 方法选择要删除的行
rows_to_drop = data.loc[start_date:end_date]
# 使用 drop() 方法删除选定的行
data = data.drop(rows_to_drop.index)
# 定义要删除的时间范围
start_date1 = '2020-06-01'
end_date1 = '2020-06-30'
# 使用 loc[] 方法选择要删除的行
rows_to_drop = data.loc[start_date1:end_date1]
# 使用 drop() 方法删除选定的行
data = data.drop(rows_to_drop.index)
# 定义要删除的时间范围
start_date2 = '2021-06-01'
end_date2 = '2021-06-30'
# 使用 loc[] 方法选择要删除的行
rows_to_drop = data.loc[start_date2:end_date2]
# 使用 drop() 方法删除选定的行
data = data.drop(rows_to_drop.index)
# 定义要删除的时间范围
start_date3 = '2021-11-01'
end_date3 = '2021-12-31'
# 使用 loc[] 方法选择要删除的行
rows_to_drop = data.loc[start_date3:end_date3]
# 使用 drop() 方法删除选定的行
data = data.drop(rows_to_drop.index)
# 定义要删除的时间范围
start_date4 = '2022-06-01'
end_date4 = '2022-06-30'
# 使用 loc[] 方法选择要删除的行
rows_to_drop = data.loc[start_date4:end_date4]
# 使用 drop() 方法删除选定的行
data = data.drop(rows_to_drop.index)
# 定义要删除的时间范围
start_date5 = '2022-11-01'
end_date5 = '2022-12-31'
# 使用 loc[] 方法选择要删除的行
rows_to_drop = data.loc[start_date5:end_date5]
# 使用 drop() 方法删除选定的行
data = data.drop(rows_to_drop.index)
ddd = np.array(data)
pdates1 = pd.date_range(start="2021-01-26", end="2023-04-27")
df = data.reindex(pdates1)
print(df.shape)
print(ddd.shape)
for i in range(len(df)):
    df[i]=ddd[i]
print(df)

###################################################################################
#print(df.index)
#print(np.array(df))
#绘制时序图
plt.plot(df.index,np.array(df))
plt.xticks(rotation=45)
plt.title("时序图")
train=np.array(df)

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

'''
ACF	 PACF	模型
拖尾	 截尾	 AR
截尾	 拖尾	 MA
拖尾	 拖尾	ARMA
如果说自相关图拖尾，并且偏自相关图在p阶截尾时，此模型应该为AR(p )。
如果说自相关图在q阶截尾并且偏自相关图拖尾时，此模型应该为MA(q)。
如果说自相关图和偏自相关图均显示为拖尾，那么可结合ACF图中最显著的阶数作为q值，选择PACF中最显著的阶数作为p值，最终建立ARMA(p,q)模型。
'''
# 进行ARIMA模型自动调参
model = auto_arima(ddd, seasonal=False, trace=False)
# 输出最优模型参数
print('最优参数为：\n',model.order)
a = model.order[0]
b = model.order[1]
c = model.order[2]
#print(model.seasonal_order)

#print(train[821])
#print(train[822])
model = sm.tsa.arima.ARIMA(train,order=(a,b,c))  #第一个是p，第三个是q
arima_res=model.fit()
arima_res.summary()
predict=arima_res.predict(823)   #前闭后闭
print(predict)

y = []
for i in range(len(ddd)):
    if ddd[i]<0:
        y.append(-1)
    elif ddd[i]==0:
        y.append(0)
    elif ddd[i]>0:
        y.append(1)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(ddd, y, test_size=0.3, random_state=0)  # train_test_split方法分割数据集
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
clf = KNeighborsClassifier()
# 训练分类器
clf.fit(X_train,y_train)

def sort(X,y,z,w):
    # 设置交叉验证次数
    n_folds = 5
    # 不同模型的名称列表
    model_names = ['clf']
    # 不同回归模型
    model_dic = [clf]
    # 交叉验证结果
    cv_score_list = []
# 各个回归模型预测的y值列表
    pre_y_list = []
# 读出每个回归模型对象
    for model in model_dic:
    # 将每个回归模型导入交叉检验
        scores = cross_val_score(model, X, y, cv=n_folds,error_score='raise')
    # 将交叉检验结果存入结果列表
        cv_score_list.append(scores)
    # 将回归训练中得到的预测y存入列表
        pre_y_list.append(model.fit(X, y).predict(X))
        if w == 1:
            print(f'{model}:',pre_y_list)
    if w == 0:
    ### 模型效果指标评估 ###
    # 获取样本量，特征数
        n_sample, n_feature = X.shape
# 分类评估指标列表
        model_metrics_list = []
# 循环每个模型的预测结果
        for pre_y in pre_y_list:
    # 临时结果列表
            tmp_list = []
        # 计算每个分类指标结果
            tmp_score = accuracy_score(y, pre_y)
        # 将结果存入临时列表
            tmp_list.append(tmp_score)
            tmp_score = recall_score(y, pre_y,average='micro')
            tmp_list.append(tmp_score)
            tmp_score = precision_score(y, pre_y,average='micro')
            tmp_list.append(tmp_score)
    #       tmp_score = average_precision_score(y, pre_y)   #二分类，只适用于两种分类的情况
    #       tmp_list.append(tmp_score)
            tmp_score = f1_score(y, pre_y,average='micro')
            tmp_list.append(tmp_score)
    # 将结果存入分类评估列表
            model_metrics_list.append(tmp_list)
        df_score = pd.DataFrame(cv_score_list, index=model_names)
        df_met = pd.DataFrame(model_metrics_list, index=model_names, columns=['准确率','召回率','精确率','F1分数'])
        print('-----------------------------',z,'-----------------------------')
# 各个交叉验证的结果,数字为MSE
        #print(df_score)
# 各种评估结果
        print(df_met)
sort(X_train,y_train,z='训练',w=0)
sort(X_test,y_test,z='测试',w=0)
y__t = clf.predict(predict.reshape(-1,1))
print('预测值：',y__t)
plt.show()
plt.close()