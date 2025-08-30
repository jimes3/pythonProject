from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,GradientBoostingClassifier
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文
plt.style.use('ggplot')
'''
'ggplot'
'bmh'
'fivethirtyeight'
'''
##########################     导入数据      #################################
df = pd.read_csv("C:\\Users\Jimes\PycharmProjects\pythonProject\study\数学建模\数据预处理\数据清洗，描述，转换\\result.csv")
print(df)
X = df.iloc[:,1:].values
y = df.iloc[:,0].values
# 标准化
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

###################    随机森林    #####################
rf = RandomForestRegressor(n_estimators=12, max_depth=6)  # 值越大，越相关
rf.fit(X,y)
print("随机森林:\n")
for i in range( X.shape[1] ):
    print (df.columns.values[i+1], format( rf.feature_importances_[i], '.9f'))
# 绘制特征重要性柱状图
plt.figure(figsize=(10, 6))
plt.barh(df.columns.values[1:], rf.feature_importances_, color='skyblue')
plt.xlabel('特征权重')
plt.title('随机森林中的特征权重')
plt.gca().invert_yaxis()  # 倒序排列，以便最重要的特征在顶部
plt.show()

###################    梯度提升树    #####################
gbc = GradientBoostingRegressor(n_estimators=100, random_state=42)  # 值越大，越相关
gbc.fit(X,y)
print("梯度提升树:\n")
for i in range( X.shape[1] ):
    print (df.columns.values[i+1], format( gbc.feature_importances_[i], '.9f'))
# 绘制特征重要性柱状图
plt.figure(figsize=(10, 6))
plt.barh(df.columns.values[1:], gbc.feature_importances_, color='skyblue')
plt.xlabel('特征权重')
plt.title('梯度提升树中的特征权重')
plt.gca().invert_yaxis()  # 倒序排列，以便最重要的特征在顶部
plt.show()