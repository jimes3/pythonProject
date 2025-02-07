import ydata_profiling as pp
import webbrowser
import warnings
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer,KNNImputer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
                 usecols=['lstat','rm', 'rad','chas'])
#数据分析
#report = df.profile_report(title='数据分析')
#report.to_file(output_file='analyse.html')
#webbrowser.open_new_tab('analyse.html')
#sns.pairplot(df,kind='reg',diag_kind='hist',hue='chas') #太慢了，不弄
#plt.savefig('关联图.png',  dpi=600)
#############################    重复数据处理     ########################
# 判断重复数据
isDuplicated = df.duplicated() # 判断重复数据记录
print('是否重复')
print (isDuplicated)

# 删除重复值
new_df1 = df.drop_duplicates() # 删除数据记录中所有列值相同的记录
#new_df2 = df.drop_duplicates(['rm']) # 删除数据记录中值相同的记录
#new_df3 = df.drop_duplicates(['rad']) # 删除数据记录中值相同的记录
#new_df4 = df.drop_duplicates(['rm', 'rad']) # 删除数据记录中指定列值相同的记录

#########################        异常值处理       #########################3
def abnormal(w=0):
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
        error_index = df[(df['rm'] <= 40) | (df['rad'] >= 12)].index.tolist()
        for i in error_index:
            df.loc[i, :] = np.nan
        #df.drop(index=error_index)
        return df
df = abnormal(w=3) # 1:z-score    2:四分位检测    3:各种值约束检测

#########################     缺失数据处理    ################################
# 查看哪些值缺失
nan_all = df.isnull() # 获得所有数据框中的N值
print('缺失值判定')
print (nan_all)
# 查看哪些列缺失
nan_col1 = df.isnull().any() # 获得含有NA的列
#nan_col2 = df.isnull().all() # 获得全部为NA的列
print ('是否存在缺失值')
print (nan_col1)
#print ('是否全缺失')
#print (nan_col2)
def fit_method(o=3):
    if o == 1:
        #简单的填充方法
        imp=SimpleImputer(strategy='mean')
        #mean：用列均值填充缺失值；median：用列中位数填充缺失值；most_frequent：使用最频繁的值填充缺失值；constant：使用指定的常数填充缺失值。
        df2=imp.fit_transform(df)
        #print('普通填充')
        #print(df2)
    if o == 2:
        #使用KNN算法填充缺失值
        imp=KNNImputer(n_neighbors=2)
        #n_neighbors的值越大，模型考虑的邻居数量也就越多，预测结果也会更加准确，但同时也会使模型计算复杂度更高。
        df2=imp.fit_transform(df)
        #print('knn填充')
        #print(df2)
    if o == 3:
        #3333丢弃缺失值
        df2 = df.dropna() # 直接丢弃含有NA的行记录
        #print('丢弃含有缺失值的行')
        #print (df2)
    return df2
df2 = fit_method(o=1)  # 1:简单填充  2:knn填充   3:丢弃不填充

####################    相关性分析        ########################
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
#  保存图片
plt.savefig('相关度.png',  dpi=600)

###########################      标准化       #######################
#X = StandardScaler().fit_transform(X_train)    #标准化
#X = MinMaxScaler().fit_transform(X_train)     #归一化

#########################    数据转换         #######################



########################      输出文件      ###########################
#定义列名列表
col_names=['lstat','rm', 'rad']
#将numpy数组转换为二维数组
data=np.atleast_2d(df2)
#将二维数组按行保存为csv文件
np.savetxt('result.csv',data,delimiter=',',fmt='%f',header=','.join(col_names),comments='')
