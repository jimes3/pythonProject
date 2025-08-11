import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   # 可视化图形调用库
import warnings
warnings.filterwarnings("ignore")  # 用于排除警告
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True)     #不以科学计数法输出
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

# 读取数据
data = pd.read_csv("实验数据.csv",index_col=0)
data1 = data
# 无量纲化
def dimensionlessProcessing(df_values, df_columns):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    res = scaler.fit_transform(df_values)
    return pd.DataFrame(res, columns=df_columns)

# 求第一列(影响因素)和其它所有列(影响因素)的灰色关联值
def GRA_ONE(data, m=0):  # m为参考列
    # 标准化
    data = dimensionlessProcessing(data.values, data.columns)
    # 参考数列
    std = data.iloc[:, m]
    # 比较数列
    ce = data.copy()
    n = ce.shape[0]
    m = ce.shape[1]
    # 与参考数列比较，相减
    grap = np.zeros([n, m])
    for i in range(m):
        for j in range(n):
            grap[j, i] = abs(ce.iloc[j, i] - std[j])
    # 取出矩阵中的最大值和最小值
    mmax = np.amax(grap)
    mmin = np.amin(grap)
    ρ = 0.5  # 灰色分辨系数
    # 计算值
    grap = pd.DataFrame(grap).applymap(lambda x: (mmin + ρ * mmax) / (x + ρ * mmax))
    # 求均值，得到灰色关联值
    RT = grap.mean(axis=0)
    return pd.Series(RT)

# 调用GRA_ONE，求得所有因素之间的灰色关联值
def GRA(data):
    list_columns = np.arange(data.shape[1])
    df_local = pd.DataFrame(columns=list_columns)
    for i in np.arange(data.shape[1]):
        df_local.iloc[:, i] = GRA_ONE(data, m=i)
    df_local.columns = list(data1)
    df_local.index=list(data1)
    return df_local

def ShowGRAHeatMap(df):
    # 色彩集
    colormap = plt.cm.RdBu
    # 设置展示一半，如果不需要注释掉mask即可
    mask = np.zeros_like(df)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(12,8))
    plt.title('Person Correlation of Features',y=1.05,size=18)
    sns.heatmap(df.astype(float),linewidths=0.1,vmax=1.0,square=True,\
               cmap=colormap,linecolor='white',annot=True,mask=mask)
    plt.xticks(rotation=45,fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.show()

data_gra = GRA(data)
print(data_gra)
ShowGRAHeatMap(data_gra)