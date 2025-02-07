import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier   # 集成算法
from sklearn.model_selection import cross_val_score    # 交叉验证
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, average_precision_score, f1_score
import ydata_profiling as pp
from matplotlib.colors import ListedColormap
import webbrowser
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("ID,crim,zn,indus,chas,nox,rm,age,di.csv",
                 usecols=['lstat','rm', 'chas'])
#pre = pd.read_csv("实验数据.csv",
#                      usecols=['lstat','rm', 'rad'])
# 数据分析
report = df.profile_report(title='数据分析')
report.to_file(output_file='analyse.html')
webbrowser.open_new_tab('analyse.html')
# 自变量
X = df[['lstat', 'rm']].values
#X_predict = pre[['lstat', 'rm']].values
# 因变量
y = df['chas'].values
#y_predict = pre['rad'].values
print('分类种类:', np.unique(y))

###################   数据集切分          ###################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  # train_test_split方法分割数据集

###############          标准化         #####################
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)    #标准化
X_test = StandardScaler().fit_transform(X_test)    #标准化
#X_predict = StandardScaler().fit_transform(X_predict)

# 可视化数据关系
sns.set(style='whitegrid', context='notebook')   #style控制默认样式,context控制着默认的画幅大小
sns.pairplot(df, size=2)
plt.savefig('数据关系.png',dpi=600)
plt.close()
# 相关度
corr = df.corr()
# 相关度热力图
sns.heatmap(corr, cmap='GnBu_r',annot=True)
plt.savefig('icon.png',dpi=600)
plt.close()
# 逻辑回归
log_model = LogisticRegression()
log_model.fit(X, y.astype('int'))
'''penalty：使用指定正则化项，可以指定为’l1’或者’l2’，L1正则化可以抵抗共线性，还会起到特征选择的作用，不重要的特征系数将会变为0；L2正则化一般不会将系数变为0，但会将不重要的特征系数变的很小，起到避免过拟合的作用。
   C：正则化强度取反，值越小正则化强度越大
   n_jobs: 指定线程数
   random_state：随机数生成器'''
# 随机森林
RFC_model = RandomForestClassifier()
RFC_model.fit(X, y.astype('int'))
'''n_estimators：森林中树的数量，默认是10棵，如果资源足够可以多设置一些。
   max_features：寻找最优分隔的最大特征数，默认是"auto"。
   max_ depth：树的最大深度。
   min_ samples_split：树中一个节点所需要用来分裂的最少样本数，默认是2。
   min_ samples_leaf：树中每个叶子节点所需要的最少的样本数。'''
# K-近邻分类
knc_model = KNeighborsClassifier()
knc_model.fit(X, y.astype('int'))
'''n_neighbors： 使用邻居的数目
   n_jobs：线程数'''
# adaboost分类
ada_model = AdaBoostClassifier()
ada_model.fit(X, y.astype('int'))
'''n_estimators： 弱分类器的数量
   learning_rate：学习率'''
# SVM
svm_model = SVC(kernel='linear', C=1.0, random_state=0).fit(X, y.astype('int'))
'''
kernel:   str      linear：线性核函数   poly：多项式核函数    rbf：径像核函数/高斯核   sigmod：sigmod核函数
c:  float    表示错误项的惩罚系数C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低；
             相反，减小C的话，容许训练样本中有一些误分类错误样本，泛化能力强。
tol:	svm停止训练的误差精度，也即阈值。	float参数 默认为1e^-3
max_iter	该参数表示最大迭代次数，如果设置为-1则表示不受限制。	int参数 默认为-1
degree	该参数只对’kernel=poly’(多项式核函数)有用，是指多项式核函数的阶数n，如果给的核函数参数是其他核函数，则会自动忽略该参数。	int型参数 默认为3
class_weight	该参数表示给每个类别分别设置不同的惩罚参数C，如果没有给，则会给所有类别都给C=1，即前面参数指出的参数C。
                如果给定参数‘balance’，则使用y的值自动调整与输入数据中的类频率成反比的权重。	字典类型或者‘balance’字符串。默认为None
'''
#梯度提升树
gbc_model = GradientBoostingClassifier().fit(X, y.astype('int'))
'''
n_estimators: 也就是弱学习器的最大迭代次数，或者说最大的弱学习器的个数。太小，容易欠拟合,太大，又容易过拟合.默认是100。
learning_rate: 即每个弱学习器的权重缩减系数?ν，也称作步长,默认1.
subsample: 子采样，取值为(0,1]，取值为1，则全部样本都使用，推荐在[0.5, 0.8]之间，默认是1.0
loss: 即我们GBDT算法中的损失函数。
max_features: 最大特征数，默认是"None"
max_depth: 决策树最大深度，默认值是3。
min_samples_split: 内部节点再划分所需最小样本数，样本量数量级非常大，则推荐增大这个值。
min_samples_leaf:  叶子节点最少样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。 默认1
min_weight_fraction_leaf： 叶子节点最小的样本权重和，叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。 默认是0
max_leaf_nodes: 最大叶子节点数，最大叶子节点数，可以防止过拟合，默认是"None”
'''

def sort(X,y,z,w):
    # 设置交叉验证次数
    n_folds = 5
    # 不同模型的名称列表
    model_names = ['log', 'rfc', 'Knc', 'adaboost','svm','gbc']
    # 不同回归模型
    model_dic = [log_model,RFC_model,knc_model,ada_model,svm_model,gbc_model]
    # 交叉验证结果
    cv_score_list = []
    # 各个回归模型预测的y值列表
    pre_y_list = []

    # 读出每个回归模型对象
    for model in model_dic:
    # 将每个回归模型导入交叉检验
        scores = cross_val_score(model, X, y, cv=n_folds,error_score='raise')
    # 将交叉检验结果存入结果列表
        cv_score_list.append(scores)
    # 将回归训练中得到的预测y存入列表
        pre_y_list.append(model.fit(X, y).predict(X))
        if w == 1:
            print(f'{model}:',pre_y_list)
    if w == 0:
        ### 模型效果指标评估 ###
        # 获取样本量，特征数
        n_sample, n_feature = X.shape
        # 分类评估指标列表
        model_metrics_list = []
        # 循环每个模型的预测结果
        for pre_y in pre_y_list:
            # 临时结果列表
            tmp_list = []
            # 计算每个分类指标结果
            tmp_score = accuracy_score(y, pre_y)
            # 将结果存入临时列表
            tmp_list.append(tmp_score)
            tmp_score = recall_score(y, pre_y,average='micro')
            tmp_list.append(tmp_score)
            tmp_score = precision_score(y, pre_y,average='micro')
            tmp_list.append(tmp_score)
            #tmp_score = average_precision_score(y, pre_y)   #二分类，只适用于两种分类的情况
            #tmp_list.append(tmp_score)
            tmp_score = f1_score(y, pre_y,average='micro')
            tmp_list.append(tmp_score)
            # 将结果存入分类评估列表
            model_metrics_list.append(tmp_list)
        df_score = pd.DataFrame(cv_score_list, index=model_names)
        df_met = pd.DataFrame(model_metrics_list, index=model_names, columns=['准确率','召回率','精确率','F1分数'])
        print('-----------------------------',z,'-----------------------------')
        # 各个交叉验证的结果,数字为MSE
        print(df_score)
        # 各种评估结果
        print(df_met)

'''#分类可视化，使用report时无用，可以用来看每一类的变量有没有什么大的差别
X_combined_std = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # 画决策边界,X是特征，y是标签，classifier是分类器，test_idx是测试集序号
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v','<','>','1','2','3','4')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 第一个特征取值范围作为横轴
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # 第二个特征取值范围作为纵轴
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))  # reolution是网格剖分粒度，xx1和xx2数组维度一样
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # classifier指定分类器，ravel是数组展平；Z的作用是对组合的二种特征进行预测
    Z = Z.reshape(xx1.shape)  # Z是列向量
    plt.contourf(xx1, xx2, Z, parameters=0.4, cmap=cmap)
    # contourf(x,y,z)其中x和y为两个等长一维数组，z为二维数组，指定每一对xy所对应的z值。
    # 对等高线间的区域进行填充（使用不同的颜色）
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    parameters=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)  # 全数据集，不同类别样本点的特征作为坐标(x,y)，用不同颜色画散点图
plot_decision_regions(X=X_combined_std, y=y_combined,classifier=gbc_model)
plt.legend(loc='upper left')
plt.tight_layout()  # 紧凑显示图片，居中显示；避免出现叠影
plt.show()
'''
sort(X_train,y_train,z='训练',w=0)
sort(X_test,y_test,z='测试',w=0)
#sort(X_redict,y_predict,z='预测',w=1)