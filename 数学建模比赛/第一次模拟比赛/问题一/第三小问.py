import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
def minmax(x):
    df = pd.read_excel("D:\.jimes\下载\第一问第三小问.xlsx",sheet_name=f'{x}')
    df = df.drop('文物采样点',axis=1)
    df=df.fillna(0)
    # 创建一个StandardScaler对象
    scaler = MinMaxScaler()
    # 对每列数据进行标准化
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled
df_scaled = minmax('高钾')
print(df_scaled)
#########第三小问
kmeans=KMeans(n_clusters=4,random_state=123,init='k-means++').fit(df_scaled)
#详细输出结果
r=pd.concat([pd.DataFrame(df_scaled), pd.Series(kmeans.labels_)], axis=1)
print(r)

df_scaled = minmax('铅钡')
print(df_scaled)
##########第三小问
kmeans=KMeans(n_clusters=4,random_state=123,init='k-means++').fit(df_scaled)
#详细输出结果
r=pd.concat([pd.DataFrame(df_scaled), pd.Series(kmeans.labels_)], axis=1)
print(r)
print(r[r[0]==0].mean())
print(r[r[0]==1].mean())
print(r[r[0]==2].mean())
print(r[r[0]==3].mean())