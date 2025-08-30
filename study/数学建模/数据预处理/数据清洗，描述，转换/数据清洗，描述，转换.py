import warnings
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
#显示所有列
pd.set_option('display.max_columns', 6)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',10)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
'''
'ggplot'
'bmh'
'fivethirtyeight'
'''

##########################     导入数据      #################################
df = pd.read_csv("C:\jimes\下载\材料一.csv")

print("=== 数据质量初步评估 ===")
print(f"数据形状: {df.shape}")  # (行数, 列数)
print("\n前5行数据:")
print(df.iloc[:,:6].head())
print("\n数值列统计描述:")
print(df.iloc[:,:6].describe())
print("\n缺失值统计:")
print(df.isnull().sum().sum())
print("\n重复行统计:", df.duplicated().sum())

#########################    数据转换         #######################
df.loc[df['励磁波形'] == '正弦波', '励磁波形'] = 1
df.loc[df['励磁波形'] == '三角波', '励磁波形'] = 2
df.loc[df['励磁波形'] == '梯形波', '励磁波形'] = 3

#############################    重复数据处理     ########################
# 删除重复值
#new_df1 = df.drop_duplicates() # 删除数据记录中所有列值相同的记录(行)
#new_df4 = df.drop_duplicates(['rm', 'rad']) # 删除数据记录中指定列值相同的记录

#########################        异常值处理       #########################3
def abnormal(w=1):
    if w==1:
        # 通过Z-Score方法判断异常值
        df_zscore = df.copy() # 复制一个用来存储Z-score得分的数据框
        cols = df.columns # 获得数据框的列名
        for col in cols: # 循环读取每列
            df_col = df[col] # 得到每列的值
            z_score = (df_col - df_col.mean()) / df_col.std() # 计算每列的Z-score得分
            df_zscore[col] = z_score.abs() > 2.2 # 判断Z-score得分是否大于2.2，如果是则是True，否则为False
        #print('是否为异常值')
        #print (df_zscore)
        columns = list(df)
        for col in columns:
            index = df_zscore[df_zscore[col] == True].index.tolist()
            for i in index:
                df.loc[i, col] = np.nan
        return df
    if w==2:
        columns = list(df)
        for col in columns:
            # 计算上下四分位数位置
            q75_bmi, q25_bmi = np.percentile(df[f'{col}'], [75, 25])
            iqr_bmi = q75_bmi - q25_bmi
            # 计算上下边界以用于异常检测
            bmi_h_bound = q75_bmi + (1.5 * iqr_bmi)
            bmi_l_bound = q25_bmi - (1.5 * iqr_bmi)
            index = df[(df[col] <= bmi_l_bound)|(df[col] >= bmi_h_bound)].index.tolist()
            for i in index:
                df.loc[i, col] = np.nan
        return df
    if w==3:
        error_index = df[(df['rm'] >= 10000) | (df['rad'] >= 12)].index.tolist()
        for i in error_index:
            df.loc[i, :] = np.nan
        return df
#df = abnormal(w=3) # 1:z-score    2:四分位检测    3:各种值约束检测

#########################     缺失数据处理    ################################
def fit_method(df2,o=3):
    if o == 1:
        #简单的填充方法
        imp=SimpleImputer(strategy='mean')
        #mean：用列均值填充缺失值；median：用列中位数填充缺失值；most_frequent：使用最频繁的值填充缺失值；constant：使用指定的常数填充缺失值。
        df2=imp.fit_transform(df)
    if o == 2:
        #使用KNN算法填充缺失值
        imp=KNNImputer(n_neighbors=2)
        #n_neighbors的值越大，模型考虑的邻居数量也就越多，预测结果也会更加准确，但同时也会使模型计算复杂度更高。
        df2=imp.fit_transform(df)
    if o == 3:
        #丢弃缺失值
        df2 = df.dropna() # 直接丢弃含有NA的行记录
    return df2
#df = fit_method(df,o=3)  # 1:简单填充  2:knn填充   3:丢弃不填充

###########################      统计指标       #######################
start_col = 4
end_col = 1029
df['均值'] = df.iloc[:,start_col:end_col].mean(axis=1)    # 左闭右开，从0开始
df['中位数'] = df.iloc[:, start_col:end_col].median(axis=1)
df['总和'] = df.iloc[:, start_col:end_col].sum(axis=1)
df['标准差'] = df.iloc[:, start_col:end_col].std(axis=1)
df['最小值'] = df.iloc[:, start_col:end_col].min(axis=1)
df['最大值'] = df.iloc[:, start_col:end_col].max(axis=1)
df['极差'] = df['最大值'] - df['最小值']
df['变异系数'] = df['标准差'] / df['均值']
# 高级统计
df['Q1'] = df.iloc[:, start_col:end_col].quantile(0.25, axis=1)
df['Q3'] = df.iloc[:, start_col:end_col].quantile(0.75, axis=1)
df['偏度'] = df.iloc[:, start_col:end_col].skew(axis=1)
df['峰度'] = df.iloc[:, start_col:end_col].kurtosis(axis=1)
# 傅里叶变换
from scipy.fft import fft, fftfreq
# 定义一个函数：输入一行序列，返回主频
def get_main_freq(row):
    T = 1 # 采样间隔
    y = row.values
    yf = fft(y)  # 复数幅值
    xf = fftfreq(len(y), T)[:len(y)//2]  # 频率值
    amplitude = 2.0/len(y) * np.abs(yf[:len(y)//2])  # 幅值
    idx = np.argmax(amplitude[1:]) + 1

    '''plt.bar(range(20), amplitude[:20])           # 用索引画
    plt.xticks(ticks=range(0, 20, 5), labels=np.round(xf[:20:5], 3))  # 把横坐标标成频率值
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()'''

    return xf[idx]
df['主频'] = df.iloc[:, start_col:end_col].apply(get_main_freq, axis=1)

# 删除原列
df = df.drop('0（磁通密度B，T）', axis=1)  # axis=1 表示作用于每一行
df = df.drop(columns=[f'{i}' for i in range(1,1024)])

# 获取列名
column_names = df.columns.tolist()
# 互换列使因变量在最前面
column_names[0],column_names[2] = column_names[2],column_names[0]

###########################      标准化       #######################
#X = StandardScaler().fit_transform(X_train)    #标准化 正态分布
#df = MinMaxScaler().fit_transform(df)     #归一化 [0,1]

########################      输出文件      ###########################
# 转换为DataFrame（无列名）
df = pd.DataFrame(df,columns=column_names)
df.to_csv('result.csv', index=False, float_format='%.6f')  # header=False