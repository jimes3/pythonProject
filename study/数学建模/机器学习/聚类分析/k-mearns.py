import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import fowlkes_mallows_score,silhouette_score,calinski_harabasz_score
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

#######################    导入数据      ######################
data = pd.read_csv("ID,crim,zn,indus,chas,nox,rm,age,di.csv",
                   usecols=['lstat','rm', 'medv'])

kmeans=KMeans(n_clusters=2,random_state=123,init='k-means++').fit(data)
'''
KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001,   
         precompute_distances='auto', verbose=0, random_state=None,  
         copy_x=True,algorithm='auto')
n-cluster	分类簇的数量
max_iter	最大的迭代次数
n_init	     算法的运行次数
init	接收待定的string。    k-means++表示该初始化策略选择的初始均值向量之间都距离比较远，它的效果较好；
                            random表示从数据中随机选择K个样本最为初始均值向量；或者提供一个数组，数组的形状为（n_cluster,n_features），该数组作为初始均值向量。
precompute_distance	   接收Boolean或者auto。表示是否提前计算好样本之间的距离，auto表示如果nsamples*n>12 million，则不提前计算。
tol	     接收float，表示算法收敛的阈值。
random_state	表示随机数生成器的种子。
verbose	0表示不输出日志信息；1表示每隔一段时间打印一次日志信息。如果大于1，打印次数频繁。
'''
###################    使用聚类来分类     #######################
result=kmeans.predict([[5.6,2.8,4.9]])
#预测的数据需要使用和训练数据同样的标准化才行。
print('预测结果:',result)

#详细输出结果
r=pd.concat([pd.DataFrame(data), pd.Series(kmeans.labels_)], axis=1)
r.columns=list(['sepal length','sepal width','petal length'])+[u'类别']
print(r)

# 可视化聚类效果
colo = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
x = data

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
shape = x.shape
sse = []
kmeans.fit(x)
sse.append(kmeans.inertia_)
lab = kmeans.fit_predict(x)
plt.xlabel('x')
plt.ylabel('y')
plt.title('3D')
centroid = kmeans.cluster_centers_
for i in range(shape[0]):
    ax.scatter(x.iloc[i,1],x.iloc[i,2], x.iloc[i,-1],c=colo[lab[i]])
ax.scatter(centroid[:,0],centroid[:,1],centroid[:,-1],marker='x',s=50,c='black')
plt.tight_layout()
plt.show()
#plt.savefig('3D.png',dpi=600)
#可视化
tsne=TSNE(n_components=2,init='random',random_state=177).fit(data)
df=pd.DataFrame(tsne.embedding_)
df['labels']=kmeans.labels_
df1=df[df['labels']==0]
df2=df[df['labels']==1]
df3=df[df['labels']==2]
#fig=plt.figure(figsize=(9,6))
centroid = kmeans.cluster_centers_
color=['red','pink','orange','gray']
fig, axi1=plt.subplots(figsize=(9,6))
axi1.scatter(centroid[:,0],centroid[:,1],marker='x',s=50,c='black')
plt.plot(df1[0],df1[1],'bo',df2[0],df2[1],'r*',df3[0],df3[1],'gD')
plt.show()


'''-----------------------------------------选择聚类的类数------------------------------------------------'''
#########################     平均离差        ######################
K = range(1, 7)
meanDispersions = []
for k in K:
    kemans = KMeans(n_clusters=k)
    kemans.fit(data)
    # 计算平均离差
    m_Disp = sum(np.min(cdist(data, kemans.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0]
    meanDispersions.append(m_Disp)
plt.plot(K, meanDispersions, 'bx-')
plt.xlabel('k')
plt.ylabel('平均离差')
plt.title('用肘部方法选择K值')
plt.show()

##########################    FMI评价法       #########################
for i in range(2,7):
    kmeans=KMeans(n_clusters=i,random_state=123).fit(data)
    score=fowlkes_mallows_score(iris_target,kmeans.labels_)
    print("聚类%d簇的FMI分数为：%f" %(i,score)) #越大越好
print('------------------------')

#########################      轮廓系数       ###########################
import matplotlib.pyplot as plt
silhouettescore=[]
for i in range(2,7):
    kmeans=KMeans(n_clusters=i,random_state=123).fit(data)
    score=silhouette_score(data, kmeans.labels_)
    silhouettescore.append(score)
plt.figure(figsize=(10,6))
plt.plot(range(2,7),silhouettescore,linewidth=1.5,linestyle='-')
plt.xlabel('k')
plt.ylabel('轮廓系数')
plt.title('用轮廓系数选择K值')
plt.show() #斜率大小

##################     Calinski-Harabasz指数评价       ###################
for i in range(2,7):
    kmeans=KMeans(n_clusters=i,random_state=123).fit(data)
    score=calinski_harabasz_score(data, kmeans.labels_)
    print("聚类%d簇的calinski_harabasz分数为：%f" %(i,score)) #越大越好

