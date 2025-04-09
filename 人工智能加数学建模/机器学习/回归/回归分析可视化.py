import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet,LogisticRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor,RandomForestClassifier,AdaBoostClassifier   # 集成算法
from sklearn.model_selection import cross_val_score    # 交叉验证
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import ydata_profiling as pp
import webbrowser
import warnings
warnings.filterwarnings("ignore")

# 数据导入
df = pd.read_csv("ID,crim,zn,indus,chas,nox,rm,age,di.csv",
                 usecols=['lstat','rm', 'medv'])
#pre = pd.read_csv("实验数据.csv",
#                      usecols=['lstat','rm', 'rad'])
# 数据分析
report = df.profile_report(title='数据分析')
report.to_file(output_file='analyse.html')
webbrowser.open_new_tab('analyse.html')
# 可视化数据关系
sns.set(style='whitegrid', context='notebook')   #style控制默认样式,context控制着默认的画幅大小
sns.pairplot(df, size=2)
plt.savefig('数据关系.png',dpi=600)
plt.close()
# 相关度
corr = df.corr()
''' method：可选值为{‘pearson’, ‘kendall’, ‘spearman’}
    pearson：Pearson相关系数来衡量两个数据集合是否在一条线上面，即针对线性数据的相关系数计算
    kendall：用于反映分类变量相关性的指标，即针对无序序列的相关系数，非正太分布的数据
    spearman：非线性的，非正太分析的数据的相关系数
    min_periods：样本最少的数据量 '''
# 相关度热力图
sns.heatmap(corr, cmap='GnBu_r',annot=True)
plt.savefig('icon.png',dpi=600)

# 自变量
X = df[['lstat', 'rm']].values
#X_predict = pre[['lstat', 'rm']].values
# 因变量
y = df['medv'].values
#y_predict = pre['rad'].values

###################   数据集切分       #####################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  # train_test_split方法分割数据集

#######################    数据标准化       ######################
X_train = StandardScaler().fit_transform(X_train)    #标准化
X_test = StandardScaler().fit_transform(X_test)    #标准化
#X = MinMaxScaler().fit_transform(X_train)     #归一化
#X_predict = StandardScaler().fit_transform(X_predict)


# 建立贝叶斯岭回归模型
br_model = BayesianRidge()
'''n_iter：迭代次数
   tol：终止迭代的阈值，即相邻两次迭代的参数差值小于tol则终止迭代
   alpha_1，alpha_2：gamma分布中参数α的形状参数和尺度参数
   lambda_1，lambda_2：gamma分布中参数λ的形状参数和尺度参数
   compute_score：是否计算每一轮迭代的模型评估得分'''
# 普通线性回归
lr_model = LinearRegression()
'''fit_intercept:是否有截据，如果没有则直线过原点。
   normalize:是否将数据归一化。
   copy_X:默认为True，当为True时，X会被copied,否则X将会被覆写。
   n_jobs:默认值为1。计算时使用的核数,-1使用所有更快'''
# 弹性网络回归模型
etc_model = ElasticNet()
'''parameters : α值。                       ρ 逐渐增大时，L1 正则项占据主导地位，代价函数越接近 Lasso 回归，当 ρ 逐渐减小时，L2 正则项占据主导地位，代价函数越接近岭回归。
   l1_ratio：ρ值，ElasticNet混合参数，其中0 <= l1_ratio <= 1。对于l1_ratio = 0，惩罚为L2范数。 对于l1_ratio = 1，为L1范数。 对于0 <l1_ratio<1，惩罚是L1和L2的组合。
   fit_intercept：一个布尔值，制定是否需要b值。
   max_iter：一个整数，指定最大迭代数。
   normalize：一个布尔值。如果为True，那么训练样本会在回归之前会被归一化。
   copy_X：一个布尔值。如果为True，会复制X，否则会覆盖X。
   precompute：一个布尔值或者一个序列。它决定是否提前计算Gram矩阵来加速计算。Gram也可以传递参数， 对于稀疏输入，此选项始终为“True”以保留稀疏性。
   tol：一个浮点数，指定判断迭代收敛与否的一个阈值。
   warm_start：一个布尔值。如果为True，那么使用前一次训练结果继续训练，否则从头开始训练。
   positive：一个布尔值。如果为True，那么强制要求权重向量的分量都为整数。
   selection：一个字符串，可以选择‘cyclic’或者‘random’。它指定了当每轮迭代的时候，选择权重向量的哪个分量来更新。
   ‘ramdom’：更新的时候，随机选择权重向量的一个分量来更新。
   ‘cyclic’：更新的时候，从前向后一次选择权重向量的一个分量来更新。
   random_state：一个整数或者一个RandomState实例，或者None。
        如果为整数，则它指定了随机数生成器的种子。
        如果为RandomState实例，则指定了随机数生成器。
        如果为None，则使用默认的随机数生成器。'''
# 支持向量机回归
svr_model = SVR()
'''kernel ： string，optional（default ='rbf'）
        指定要在算法中使用的内核类型。它必须是'linear'，'poly'，'rbf'，'sigmoid'，'precomputed'或者callable之一。如果没有给出，将使用'rbf'。如果给出了callable，则它用于预先计算内核矩阵。
   degree： int，可选（默认= 3）   多项式核函数的次数（'poly'）。被所有其他内核忽略。
   gamma ： float，optional（默认='auto'）   'rbf'，'poly'和'sigmoid'的核系数。
        当前默认值为'auto'，它使用1 / n_features，如果gamma='step'传递，则使用1 /（n_features * X.std（））作为gamma的值。当前默认的gamma''auto'将在版本0.22中更改为'step'。'auto_deprecated'，'auto'的弃用版本用作默认值，表示没有传递明确的gamma值。
   coef0 ： float，optional（默认值= 0.0）  核函数中的独立项。它只在'poly'和'sigmoid'中很重要。
   tol ： float，optional（默认值= 1e-3）容忍停止标准。
   C ： float，可选（默认= 1.0）   错误术语的惩罚参数C.
   epsilon ： float，optional（默认值= 0.1）
        Epsilon在epsilon-SVR模型中。它指定了epsilon-tube，其中训练损失函数中没有惩罚与在实际值的距离epsilon内预测的点。
   shrinking ： 布尔值，可选（默认= True）  是否使用收缩启发式。
   cache_size ： float，可选   指定内核缓存的大小（以MB为单位）。
   verbose ： bool，默认值：False
        启用详细输出。请注意，此设置利用libsvm中的每进程运行时设置，如果启用，则可能无法在多线程上下文中正常运行。
   max_iter ： int，optional（默认值= -1）   求解器内迭代的硬限制，或无限制的-1'''
# 梯度增强回归模型对象
gbr_model = GradientBoostingRegressor()
'''1) 划分时考虑的最大特征数max_features: 可以使用很多种类型的值，默认是"None",意味着划分时考虑所有的特征数；如果是"log2"意味着划分时最多考虑log2N个特征；如果是"sqrt"或者"auto"意味着划分时最多考虑N−−√个特征。
        如果是整数，代表考虑的特征绝对数。如果是浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。其中N为样本总特征数。一般来说，如果样本特征数不多，比如小于50，我们用默认的"None"就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。
   2) 决策树最大深度max_depth: 默认可以不输入，如果不输入的话，默认值是3。一般来说，数据少或者特征少的时候可以不管这个值。如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。
   3) 内部节点再划分所需最小样本数min_samples_split: 这个值限制了子树继续划分的条件，如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分。 默认是2.如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。
   4) 叶子节点最少样本数min_samples_leaf: 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。 默认是1,可以输入最少的样本数的整数，或者最少样本数占样本总数的百分比。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。
   5）叶子节点最小的样本权重和min_weight_fraction_leaf：这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。 默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。
   6) 最大叶子节点数max_leaf_nodes: 通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。
   7) 节点划分最小不纯度min_impurity_split:  这个值限制了决策树的增长，如果某节点的不纯度(基于基尼系数，均方差)小于这个阈值，则该节点不再生成子节点。即为叶子节点 。一般不推荐改动默认值1e-7。'''

def regressor(X,y,z,w):
    # 设置交叉验证次数
    n_folds = 5
    # 不同模型的名称列表
    model_names = ['BayesianRidge', 'LinearRegression', 'ElasticNet', 'SVR', 'GBR']
    # 不同回归模型
    model_dic = [br_model, lr_model, etc_model, svr_model, gbr_model]
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
    ### 模型效果指标评估 ###
    # 获取样本量，特征数
    n_sample, n_feature = X.shape
    # 回归评估指标对象列表
    model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]
    # 回归评估指标列表
    model_metrics_list = []
    # 循环每个模型的预测结果
    for pre_y in pre_y_list:
        # 临时结果列表
        tmp_list = []
        # 循环每个指标对象
        for mdl in model_metrics_name:
            # 计算每个回归指标结果
            tmp_score = mdl(y, pre_y)
            # 将结果存入临时列表
            tmp_list.append(tmp_score)

        # 将结果存入回归评估列表
        model_metrics_list.append(tmp_list)
    df_score = pd.DataFrame(cv_score_list, index=model_names)
    df_met = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])

    print('----------------------------------',z,'-----------------------------------')
    # 各个交叉验证的结果,数字为MSE
    print(df_score)
    # 各种评估结果
    print(df_met)
    '''
    ev     explained_variance_score：解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量的方差变化，值越小则说明效果越差。
    mae    mean_absolute_error：平均绝对误差（Mean Absolute Error, MAE），用于评估预测结果和真实数据集的接近程度的程度，其值越小说明拟合效果越好。
    mse    mean_squared_error：均方差（Mean squared error, MSE），该指标计算的是拟合数据和原始数据对应样本点的误差的平方和的均值，其值越小说明拟合效果越好。
    r2     r2_score：判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量的方差变化，值越小则说明效果越差。
    '''
    ### 可视化 ###
    # 创建画布
    plt.figure(figsize=(9, 6))
    # 颜色列表
    color_list = ['r', 'g', 'b', 'y', 'c']
    # 循环结果画图
    for i, pre_y in enumerate(pre_y_list):
        # 子网络
        plt.subplot(2, 3, i+1)
        # 画出原始值的曲线
        plt.plot(np.arange(X.shape[0]), y, color='k', label='y')
        # 画出各个模型的预测线
        plt.plot(np.arange(X.shape[0]), pre_y, color_list[i], label=model_names[i])
        plt.title(model_names[i])
        plt.legend(loc='lower left')
    plt.savefig(f'{z}现实对比.png',dpi=600)
    plt.show()

regressor(X_train,y_train,z='训练',w=0)
regressor(X_test,y_test,z='测试',w=0)
#regressor(X_predict,y_predict,z='预测',w=1)