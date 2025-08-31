#import ydata_profiling as pp
import pandas as pd
import webbrowser
import seaborn as sns
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
#df = pd.read_csv("C:\jimes\下载\材料一.csv")
df = pd.read_csv("C:\\Users\Jimes\PycharmProjects\pythonProject\study\数学建模\数据预处理\数据清洗，描述，转换\\result.csv")

# 计算均值,并添加一行
#mean_row = df.iloc[:,5:1000].mean()
#df.loc["mean"] = mean_row

row_data = df.iloc[2000, 5:1000]

plt.figure(figsize=(10, 6))
plt.plot(range(len(row_data)), row_data.values, marker='o', linewidth=2, markersize=8)
plt.title('数据')
plt.xlabel('时间')
plt.ylabel('数值')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#数据分析
'''report = df.profile_report(title='数据分析')
report.to_file(output_file='analyse.html')
webbrowser.open_new_tab('analyse.html')
sns.pairplot(df,kind='reg',diag_kind='hist',hue='chas') #太慢了，不弄
plt.savefig('关联图.png',  dpi=600)'''


####################    相关性分析        ########################
#  选择计算相关系数的方法
corr1  =  df.corr(method='pearson',  min_periods=10)   # 线性相关、连续、服从正态分布的数据集。min_periods，最小计算需求数量值
corr2  =  df.corr(method='kendall',  min_periods=10)  #皮尔逊Pearson相关系数使用前提条件中，任何一个条件不满足时可以考虑使用该系数，建议数据大于500
corr3  =  df.corr(method='spearman',  min_periods=10)  #衡量有序分类型数据的序数相关性
#  绘制相关度热力图
fig = plt.figure(figsize=(20, 20))
sns.heatmap(corr2,  cmap='GnBu_r',  annot=True)
plt.title('相关性分析')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
#plt.savefig('相关度.png',  dpi=600)
plt.show()