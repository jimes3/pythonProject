import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,GradientBoostingClassifier
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文
plt.style.use('ggplot')

##########################     导入数据      #################################
df = pd.read_csv("3.时域频域特征.csv")
X = df.iloc[:,1:].values
y = df.iloc[:,0].values

scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)   # 特征标准化
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1,1))   # 标签标准化

###################    随机森林    #####################
rf = RandomForestRegressor(n_estimators=12, max_depth=6)  # 值越大，越相关
rf.fit(X,y)
print("随机森林:\n")
for i in range( X.shape[1] ):
    print (df.columns.values[i+1], format( rf.feature_importances_[i], '.9f'))
# 绘制特征重要性柱状图
features = df.columns.values[1:]
# 按权重大小排序
sorted_idx = np.argsort(rf.feature_importances_)[::-1]  # 从大到小
sorted_features = [features[i] for i in sorted_idx]
sorted_weights = [rf.feature_importances_[i] for i in sorted_idx]
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_weights, color='skyblue')
plt.xlabel('特征权重')
plt.title('随机森林中的特征权重')
plt.show()

###################    梯度提升树    #####################
gbc = GradientBoostingRegressor(n_estimators=100, random_state=42)  # 值越大，越相关
gbc.fit(X,y)
print("梯度提升树:\n")
for i in range( X.shape[1] ):
    print (df.columns.values[i+1], format( gbc.feature_importances_[i], '.9f'))
# 绘制特征重要性柱状图
features = df.columns.values[1:]
# 按权重大小排序
sorted_idx = np.argsort(gbc.feature_importances_)[::-1]  # 从大到小
sorted_features = [features[i] for i in sorted_idx]
sorted_weights = [gbc.feature_importances_[i] for i in sorted_idx]
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_weights, color='skyblue')
plt.xlabel('特征权重')
plt.title('梯度提升树中的特征权重')
plt.show()