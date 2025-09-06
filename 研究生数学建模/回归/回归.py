from lce import LCERegressor
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tqdm import tqdm
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文
plt.style.use('ggplot')

# ---------------- 数据读取 ----------------
df = pd.read_csv("../数据预处理/3.时域频域特征.csv")
features = df.columns[1:].tolist()
y = df['磁芯损耗，w/m3'].values
X = df[features].values  # numpy格式

# ---------------- 数据集切分 ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# 特征标准化
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)   # 用训练集fit
X_test = scaler_X.transform(X_test)         # 测试集用训练集参数

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1,1))
y_test = scaler_y.transform(y_test.reshape(-1,1))

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
# 随机森林
rf_model = RandomForestRegressor(
    n_estimators=100,       # 树的数量，默认100
    max_depth=None,         # 树的最大深度，None表示不限制
    min_samples_split=2,    # 内部节点再划分所需最小样本数
    min_samples_leaf=1,     # 叶子节点最少样本数
    max_features='auto',    # 每次分裂考虑的最大特征数，auto=√特征数
    bootstrap=True,         # 是否有放回抽样
    random_state=42,        # 随机种子，保证可复现
    n_jobs=-1               # 并行计算，-1使用所有CPU
)
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

#  极端梯度增强随机森林
lce_model = LCERegressor(
    n_estimators=10,
    bootstrap=True,
    max_samples=0.8,
    max_features="sqrt",
    max_depth=3,
    min_samples_leaf=1,
    metric="neg_mean_squared_error",   # 回归指标
    n_iter=1,
    base_learner="xgboost",
    base_n_estimators=(100,),
    base_max_depth=(3,),
    base_learning_rate=(1,),
    base_gamma=(0,),
    base_min_child_weight=(1,),
    base_subsample=(1.0,),
    base_colsample_bytree=(1.0,),
    base_reg_alpha=(0,),
    base_reg_lambda=(0,),
    base_booster=("gbtree",),
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# ---------------- 评估函数 ----------------
def evaluate_regressor(model, X, y, w='', X_test=None):
    if w == "预测":
        # 用完整训练集训练模型预测新数据
        y_pred = model.fit(X, y).predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1))
        print('-----------------------------预测-----------------------------')
        print('模型预测结果:', [round(i,4) for i in y_pred.ravel()])
    if w == "稳定":
        n_folds = 3
        print("开始交叉验证")
        # 交叉验证评分
        scores = cross_val_score(model, X, y, cv=n_folds, scoring='neg_mean_squared_error')
        # 交叉验证预测结果
        y_pred = cross_val_predict(model, X, y, cv=n_folds)
        # 回归评估指标
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        # 输出结果
        df_score = pd.DataFrame([-scores], columns=[f'Fold{i+1}' for i in range(n_folds)])  # 取负号得到 MSE
        df_metrics = pd.DataFrame([[round(mse,4), round(rmse,4), round(mae,4), round(r2,4)]],
                                  columns=['MSE', 'RMSE', 'MAE', 'R²'])
        print('-----------------------------训练-----------------------------')
        print('交叉验证MSE:')
        print(df_score)
        print(f"Mean CV MSE: {(-scores).mean():.4f}")
        print(f"Std CV MSE: {scores.std():.4f}")
        print('性能评估指标:')
        print(df_metrics)
    if w == "训练":
        n_folds = 3
        # 不同模型的名称列表
        model_names = ['lr', 'rf', 'svr', 'gbc','lce']
        # 不同回归模型
        model_dic = [lr_model,rf_model,svr_model,gbr_model,lce_model]
        # 交叉验证结果
        cv_score_list = []
        # 各个模型预测的y值列表
        pre_y_list = []
        for model in tqdm(model_dic):
            # 将每个模型导入交叉检验
            scores = cross_val_score(model, X, y, cv=n_folds, scoring='neg_mean_squared_error')
            # 将交叉检验结果存入结果列表
            cv_score_list.append(-scores)
            # 将训练中得到的预测y存入列表
            pre_y_list.append(cross_val_predict(model, X, y, cv=n_folds))
        ### 模型效果指标评估 ###
        # 分类评估指标列表
        model_metrics_list = []
        # 循环每个模型的预测结果
        for y_pred in pre_y_list:
            # 临时结果列表
            tmp_list = []
            # 计算每个分类指标结果
            mse = mean_squared_error(y, y_pred)
            tmp_list.append(mse)
            tmp_list.append(np.sqrt(mse))
            tmp_list.append(mean_absolute_error(y, y_pred))
            tmp_list.append(r2_score(y, y_pred))
            # 将结果存入分类评估列表
            model_metrics_list.append(tmp_list)
        df_score = pd.DataFrame(cv_score_list, index=model_names)
        # 计算每行的均值和标准差
        df_score['均值'] = df_score.mean(axis=1)
        df_score['方差'] = df_score.std(axis=1)
        df_metrics = pd.DataFrame(model_metrics_list, index=model_names,columns=['MSE', 'RMSE', 'MAE', 'R²'])
        print('-----------------------------训练-----------------------------')
        print('交叉验证MSE:')
        print(df_score)
        print(df_metrics)
        ### 可视化 ###
        # 创建画布
        plt.figure(figsize=(9, 6))
        # 颜色列表
        color_list = ['r', 'g', 'b', 'y', 'c']
        # 循环结果画图
        for i, pre_y in enumerate(pre_y_list):
            # 子网络
            plt.subplot(3, 2, i+1)
            # 画出原始值的曲线
            plt.plot(np.arange(X.shape[0]), y, color='k', label='y')
            # 画出各个模型的预测线
            plt.plot(np.arange(X.shape[0]), pre_y, color_list[i], label=model_names[i])
            plt.title(model_names[i])
            plt.legend(loc='lower left')
        plt.savefig('测试现实对比.png',dpi=3600)
        plt.show()
    if w == "残差分析":   # 残差必须要近似正态
        y_pred = model.fit(X, y).predict(X)
        # 残差分析
        residuals = y - y_pred
        # 对 residuals 随机采样，否则时间太长
        sample_resid = np.random.choice(residuals.ravel(), size=1000, replace=True)
        sns.histplot(sample_resid, kde=True)
        plt.title("残差正态分布检验")
        plt.show()
        import statsmodels.api as sm
        sm.qqplot(sample_resid, line='45', fit=True)
        plt.title("残差QQ图")  #靠近45度线表明符合正态
        plt.show()
        from scipy.stats import shapiro
        stat, p = shapiro(residuals)
        print('Shapiro-Wilk test p-value:', p)
        if p > 0.05:
            print("残差近似正态")
        else:
            print("残差偏离正态")
        # 残差标准差
        n, p = len(y), 1
        sigma = np.sqrt(np.sum(residuals**2) / (n - p))
        # 置信区间
        alpha = 0.06
        t_val = stats.t.ppf(1 - alpha/2, df=n - p)
        ci_lower = y_pred - t_val * sigma
        ci_upper = y_pred + t_val * sigma
        in_ci = (y >= ci_lower) & (y <= ci_upper)
        coverage = np.mean(in_ci)
        print("置信区间覆盖率:", coverage)
        plt.ioff()
        # 按大小排序方便可视化
        sort_idx = np.argsort(y_pred)
        y_sorted = y[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        ci_lower_sorted = ci_lower[sort_idx]
        ci_upper_sorted = ci_upper[sort_idx]
        # 可视化
        plt.scatter(range(len(y)), y_sorted, label="Data")
        plt.plot(range(len(y_pred)), y_pred_sorted, color="red", label="Fitted curve")
        plt.fill_between(range(len(y)), ci_lower_sorted, ci_upper_sorted, color="pink", alpha=0.3, label="95% CI")
        plt.legend()
        plt.show()

# 敏感性分析
def mingan():
    import shap
    lce_model.fit(X_train, y_train)
    explainer = shap.KernelExplainer(lce_model.predict,shap.sample(X_train, 100))
    shap_values = explainer.shap_values(X_test[:3])
    # 计算全局平均SHAP绝对值
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    print('每个特征的贡献占比：\n',mean_abs_shap/sum(mean_abs_shap))  # 每个特征的贡献
    # 绘制特征重要性柱状图，按权重大小排序
    sorted_idx = np.argsort(mean_abs_shap)[::-1]  # 从大到小
    sorted_features = [features[i] for i in sorted_idx]
    sorted_weights = [mean_abs_shap[i] for i in sorted_idx]
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_weights, color='skyblue')
    plt.xlabel('特征权重')
    plt.title('SHAP分析后的特征权重')
    plt.show()
    plt.ioff()

# 稳定性分析
def wending():
    noise_level = 0.05  # 设置噪声强度，可以调节大小
    X_train_noisy = X_train + np.random.normal(loc=0.0, scale=noise_level, size=X_train.shape)
    # 交叉验证训练
    evaluate_regressor(lce_model, X_train_noisy, y_train, w="稳定")

# 不确定性分析
def buqueding():
    evaluate_regressor(lce_model, X_train, y_train, w="残差分析")
    # 残差不符合正态分布，使用Bootstrap
    from sklearn.utils import resample
    y_preds = []
    for i in tqdm(range(10)): # Bootstrap重复次数
        # 有放回采样训练集
        X_resampled, y_resampled = resample(X_train, y_train, random_state=i)
        # 拟合模型
        lce_model.fit(X_resampled, y_resampled)
        # 对测试集预测
        y_pred = lce_model.predict(X_test)
        y_preds.append(y_pred)
    y_preds = np.array(y_preds)
    # 计算均值和标准差
    y_mean = y_preds.mean(axis=0)
    # 95% 置信区间
    lower = np.percentile(y_preds, 2.5, axis=0)
    upper = np.percentile(y_preds, 97.5, axis=0)
    # 为了画图，按 X 排序
    order = np.argsort(y_mean)  # 以第一维特征为横轴
    y_mean_plot = y_mean[order]
    lower_plot = lower[order]
    upper_plot = upper[order]
    # 绘图
    plt.figure(figsize=(8,5))
    plt.plot(range(len(y_mean_plot)), y_mean_plot, color="blue", label="预测均值")
    plt.fill_between(range(len(y_mean_plot)), lower_plot, upper_plot, color="blue", alpha=0.2, label="95% CI")
    plt.xlabel("X")
    plt.ylabel("预测值")
    plt.title("Bootstrap 预测均值与置信区间")
    plt.legend()
    plt.show()

# 交叉验证训练
#evaluate_regressor(lce_model, X_train, y_train, w="训练")
# 用训练好的模型预测测试集
#evaluate_regressor(lce_model, X_train, y_train, w="预测", X_test=X_test)
print('---------------------敏感性分析---------------------')
mingan()
print('---------------------稳定性分析---------------------')
wending()
print('---------------------不确定性分析---------------------')
buqueding()