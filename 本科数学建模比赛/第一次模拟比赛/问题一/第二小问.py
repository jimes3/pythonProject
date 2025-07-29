import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文


##########第二小问
def jianyan():
    df = pd.read_excel("D:\.jimes\下载\C题\附件.xlsx",sheet_name = '表单2')
    df = df.drop('文物采样点',axis=1)
    df=df.fillna(0)
    re = df.sum(axis=1)
    for i in range(len(re)):
        if re[i]<85 or re[i]>105:
            print('采样点excel行号：',i+2,re[i])
    df = df.drop([17,19])
    return df
def minmax(x):
    df = pd.read_excel("D:\.jimes\下载\第一问第二小问.xlsx",sheet_name=f'{x}')
    df = df.drop('文物采样点',axis=1)
    df=df.fillna(0)
    # 创建一个StandardScaler对象
    scaler = MinMaxScaler()
    # 对每列数据进行标准化
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled


fig,axes = plt.subplots(2,2,figsize=(9,6))

df_scaled = minmax('高钾风化')
print(df_scaled.describe(include='all'))
kurtosis_values = df_scaled.kurtosis()
skewness_values = df_scaled.skew()
coeff_of_variation = df_scaled.std() / df_scaled.mean() * 100
print("峰度系数:",kurtosis_values)
print("偏度系数:",skewness_values)
print("变异系数:",coeff_of_variation)
df_scaled.plot.box(ax=axes[0, 0],title="高钾风化",patch_artist=True, boxprops={'facecolor': 'red'},
                   medianprops={'linewidth': 2, 'color': 'black'},showmeans=True)
axes[0, 0].grid(linestyle="--", alpha=0.1)
axes[0, 0].tick_params(axis='x', rotation=30)

df_scaled = minmax('高钾无风化')
print(df_scaled.describe(include='all'))
kurtosis_values = df_scaled.kurtosis()
skewness_values = df_scaled.skew()
coeff_of_variation = df_scaled.std() / df_scaled.mean() * 100
print("峰度系数:",kurtosis_values)
print("偏度系数:",skewness_values)
print("变异系数:",coeff_of_variation)
df_scaled.plot.box(ax=axes[0, 1],title="高钾无风化",patch_artist=True, boxprops={'facecolor': 'red'},
                   medianprops={'linewidth': 2, 'color': 'black'},showmeans=True)
axes[0, 1].grid(linestyle="--", alpha=0.1)
axes[0, 1].tick_params(axis='x', rotation=30)

df_scaled = minmax('铅钡风化')
print(df_scaled.describe(include='all'))
kurtosis_values = df_scaled.kurtosis()
skewness_values = df_scaled.skew()
coeff_of_variation = df_scaled.std() / df_scaled.mean() * 100
print("峰度系数:",kurtosis_values)
print("偏度系数:",skewness_values)
print("变异系数:",coeff_of_variation)
df_scaled.plot.box(ax=axes[1, 0],title="铅钡风化",patch_artist=True, boxprops={'facecolor': 'red'},
                   medianprops={'linewidth': 2, 'color': 'black'},showmeans=True)
axes[1, 0].grid(linestyle="--", alpha=0.1)
axes[1, 0].tick_params(axis='x', rotation=30)

df_scaled = minmax('铅钡无风化')
print(df_scaled.describe(include='all'))
kurtosis_values = df_scaled.kurtosis()
skewness_values = df_scaled.skew()
coeff_of_variation = df_scaled.std() / df_scaled.mean() * 100
print("峰度系数:",kurtosis_values)
print("偏度系数:",skewness_values)
print("变异系数:",coeff_of_variation)
df_scaled.plot.box(ax=axes[1, 1],title="铅钡无风化",patch_artist=True, boxprops={'facecolor': 'red'},
                   medianprops={'linewidth': 2, 'color': 'black'},showmeans=True)
axes[1, 1].grid(linestyle="--", alpha=0.1)
axes[1, 1].tick_params(axis='x', rotation=30)
# 调整子图之间的距离
plt.tight_layout()
plt.show()
