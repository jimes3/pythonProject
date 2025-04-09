import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston,load_iris
from sklearn.feature_selection import VarianceThreshold,mutual_info_classif,chi2, SelectKBest
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

##########################     导入数据      #################################
df = pd.read_csv("ID,crim,zn,indus,chas,nox,rm,age,di.csv",
                 usecols=['crim','age', 'medv','dis'])
X = df[['crim', 'age','dis']].values
y = df['medv'].values

###################      方差选择       ####################
# 查看数据标准差
print('数据标准差:\n',df.std())

# 利用sklearn包进行方差选择
vt = VarianceThreshold(threshold=4)  #当被导入特征的'方差'小于threshold值时，该特征将会被过滤,注意上面是标准差
xx = vt.fit_transform(df)
#print(xx.shape)  #查看处理后的行列数

#######################       相关系数法         ########################
#  选择计算相关系数的方法
corr1  =  df.corr(method='pearson',  min_periods=10)   # 线性相关、连续、服从正态分布的数据集。min_periods，最小计算需求数量值
corr2  =  df.corr(method='kendall',  min_periods=10)  #皮尔逊Pearson相关系数使用前提条件中，任何一个条件不满足时可以考虑使用该系数，建议数据大于500
corr3  =  df.corr(method='spearman',  min_periods=10)  #衡量有序分类型数据的序数相关性
#  绘制相关度热力图
fig,  axs  =  plt.subplots(2,  2,  figsize=(10,  8))
axs[0,  0].set_title('Pearson  Coefficient')
sns.heatmap(corr1,  cmap='GnBu_r',  annot=True,  ax=axs[0,  0])
axs[0,  1].set_title('Kendall  Coefficient')
sns.heatmap(corr2,  cmap='GnBu_r',  annot=True,  ax=axs[0,  1])
axs[1,  0].set_title('Spearman  Coefficient')
sns.heatmap(corr3,  cmap='GnBu_r',  annot=True,  ax=axs[1,  0])
plt.show()
plt.close()

######################     卡方检验      ########################
X, y = load_iris(return_X_y=True)
chi2_model = SelectKBest(chi2, k=3)
chi2_model.fit_transform(X, y)
print('卡方检验：\n------------评分--------------p值--------------')
for i in range(X.shape[1]):
    print((chi2_model.scores_[i],'%.9f' % chi2_model.pvalues_[i]))
    #p值为置信度,评分越大越相关。不同自由度下最低评分要求不同，自由度=（行-1）*（列-1）

####################    互信息法        ########################
X_mut = mutual_info_classif(X, y)
print('  互信息法：\n',X_mut)
#越大越相关，适用于离散数据

''' 
    方差选择法：只适用于连续变量
    皮尔逊相关系数：适用于特征类型均为数值特征的情况
    卡方检验：只适用于连续变量
    互信息法：它不属于度量方式，也没有办法归一化，在不同数据集上的结果无法做比较。对于连续变量通常需要先离散化，而互信息的结果对离散化的方式敏感。
'''