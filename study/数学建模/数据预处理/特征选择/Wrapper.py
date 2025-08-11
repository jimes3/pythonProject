from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
##########################     导入数据      #################################
df = pd.read_csv("ID,crim,zn,indus,chas,nox,rm,age,di.csv",
                 usecols=['crim','age', 'medv','rad'])
X = df[['crim', 'age','rad']].values
y = df['medv'].values


rfe = RFE(estimator=LogisticRegression(max_iter=3000), n_features_to_select=2).fit(X, y.astype('int'))
print(rfe.support_, rfe.ranking_)
'''
1,指定一个有n个特征的数据集。
2,选择一个算法模型来做RFE的基模型。
3,指定保留的特征数量 k(k<n)。
4,第一轮对所有特征进行训练，算法会根据基模型的目标函数给出每个特征的 “得分”或排名，
    将最小“得分”或排名的特征剔除，这时候特征减少为n-1，对其进行第二轮训练，持续迭代，直到特征保留为k个，这k个特征就是选择的特征。
'''