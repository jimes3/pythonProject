import ydata_profiling as pp
import webbrowser
import warnings
warnings.filterwarnings("ignore")
from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pylab
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

o = 2
if o==0:
    #数据分析
    report = df.profile_report(title='数据分析')
    report.to_file(output_file='analyse.html')
    webbrowser.open_new_tab('analyse.html')
list_FAHUO = ['A','B','C','D','E','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
list_SHOUHUO = ['A','B','C','D','E','G','H','I','J','K','L','M','N','O','P','Q','R','S','U','V','W','X','Y']

def fahuo(df,i):
    df = df.loc[df['发货城市 (Delivering city)']==list_FAHUO[i]]
    df = df.resample("B").sum()
    pdates = pd.date_range(start="2018-04-19", end="2019-04-17")
    df = df.reindex(pdates, fill_value=-1)
    # 将缺失值标记为np.nan
    df.replace(-1, np.nan, inplace=True)
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
    imp = KNNImputer(n_neighbors=5)
    # n_neighbors的值越大，模型考虑的邻居数量也就越多，预测结果也会更加准确，但同时也会使模型计算复杂度更高。
    df = imp.fit_transform(df)
    # 定义要删除的时间范围
    start_date = '2018-11-01'
    end_date = '2018-12-31'
    # 使用 loc[] 方法选择要删除的行
    df = pd.DataFrame(df)
    rows_to_drop = df.loc[start_date:end_date]
    # 使用 drop() 方法删除选定的行
    df = df.drop(rows_to_drop.index)
    # 定义要删除的时间范围
    start_date1 = '2018-06-01'
    end_date1 = '2018-06-30'
    # 使用 loc[] 方法选择要删除的行
    rows_to_drop = df.loc[start_date1:end_date1]
    # 使用 drop() 方法删除选定的行
    data = df.drop(rows_to_drop.index)
    data = np.array(data)
    return data
def shouhuo(df,i):
    df = df.loc[df['收货城市 (Receiving city)']==list_SHOUHUO[i]]
    df = df.resample("B").sum()
    pdates = pd.date_range(start="2018-04-19", end="2019-04-17")
    df = df.reindex(pdates, fill_value=-1)
    # 将缺失值标记为np.nan
    df.replace(-1, np.nan, inplace=True)
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
    imp = KNNImputer(n_neighbors=5)
    # n_neighbors的值越大，模型考虑的邻居数量也就越多，预测结果也会更加准确，但同时也会使模型计算复杂度更高。
    df = imp.fit_transform(df)
    # 定义要删除的时间范围
    start_date = '2018-11-01'
    end_date = '2018-12-31'
    # 使用 loc[] 方法选择要删除的行
    df = pd.DataFrame(df)
    rows_to_drop = df.loc[start_date:end_date]
    # 使用 drop() 方法删除选定的行
    df = df.drop(rows_to_drop.index)
    # 定义要删除的时间范围
    start_date1 = '2018-06-01'
    end_date1 = '2018-06-30'
    # 使用 loc[] 方法选择要删除的行
    rows_to_drop = df.loc[start_date1:end_date1]
    # 使用 drop() 方法删除选定的行
    data = df.drop(rows_to_drop.index)
    data = np.array(data)
    return data

createv = locals()
myVarList = [] # 存放自己创建的变量
for i in range(len(list_SHOUHUO)):
    createv['shouhuo'+ str(list_SHOUHUO[i])] = shouhuo(df,i)
    myVarList.append(createv['shouhuo' + str(list_SHOUHUO[i])])
myVarList1 = []  # 存放自己创建的变量
for i in range(len(list_FAHUO)):
    createv['fahuo' + str(list_FAHUO[i])] = fahuo(df, i)
    myVarList1.append(createv['fahuo' + str(list_FAHUO[i])])


def func(x, a, b):
    return a * x+b
for v in range(len(list_SHOUHUO)):
    x = [i for i in range(len(shouhuoA))]
    y = myVarList[v].ravel()
    popt, pcov = curve_fit(func, x, y)
    y_pred = [func(i, popt[0], popt[1]) for i in x]
    print('系数:',popt[0])

for v in range(len(list_FAHUO)):
    x = [i for i in range(len(fahuoA))]
    y = myVarList1[v].ravel()
    popt, pcov = curve_fit(func, x, y)
    y_pred = [func(i, popt[0], popt[1]) for i in x]
    print('系数:',popt[0])

zonga = fahuoA+shouhuoA
zongb = fahuoB+shouhuoB
zongc = fahuoC+shouhuoC
zongd = fahuoD+shouhuoD
zonge = fahuoE+shouhuoE
zongg = fahuoG+shouhuoG
zongh = fahuoH+shouhuoH
zongi = fahuoI+shouhuoI
zongj = fahuoJ+shouhuoJ
zongk = fahuoK+shouhuoK
zongl = fahuoL+shouhuoL
zongm = fahuoM+shouhuoM
zongn = fahuoN+shouhuoN
zongo = fahuoO+shouhuoO
zongp = fahuoP+shouhuoP
zongq = fahuoQ+shouhuoQ
zongr = fahuoR+shouhuoR
zongs = fahuoS+shouhuoS
zongt = fahuoT
zongu = fahuoU+shouhuoU
zongv = fahuoV+shouhuoV
zongw = fahuoW+shouhuoW
zongx = fahuoX+shouhuoX
zongy = fahuoY+shouhuoY
hh = np.vstack((zonga.ravel(),zongb.ravel(),zongc.ravel(),zongd.ravel(),zonge.ravel(),zongg.ravel(),zongh.ravel(),zongi.ravel()
                ,zongj.ravel(),zongk.ravel(),zongl.ravel(),zongm.ravel(),zongn.ravel(),zongo.ravel(),zongp.ravel(),zongq.ravel()
                ,zongr.ravel(),zongs.ravel(),zongt.ravel(),zongu.ravel(),zongv.ravel(),zongw.ravel(),zongx.ravel(),zongy.ravel()))
#zong = np.sum(hh, axis =1 )
zong_pd = pd.DataFrame(hh.T)
print(zong_pd)
data = zong_pd
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
    #mask = np.zeros_like(df)
    #mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(12,12))
    plt.title('Correlation of Features',y=1,size=13)
    sns.heatmap(df.astype(float),linewidths=0.1,vmax=1.0,square=True,\
               cmap=colormap,linecolor='white',annot=True,annot_kws={"fontsize":7, "color":"black"})
    plt.xlim(0, None)
    #plt.ylim(0, None)
    plt.xticks(range(0,24,1),labels=['A','B','C','D','E','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y'],
               rotation=45,fontsize=10)
    plt.yticks(range(0,24,1),labels=['A','B','C','D','E','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y'],
               rotation=0, fontsize=10)
    plt.show()


data_gra = GRA(data)
print(data_gra)
ShowGRAHeatMap(data_gra)
print(data_gra.mean())
