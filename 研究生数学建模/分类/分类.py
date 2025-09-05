import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier   # 集成算法
from sklearn.model_selection import cross_val_score,cross_val_predict    # 交叉验证
from tqdm import tqdm
from lce import LCEClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文
plt.style.use('ggplot')

df = pd.read_csv("../数据预处理/3.时域频域特征.csv")
features = df.columns[4:].tolist()
y = df['励磁波形'].values
X = df.iloc[:, 4:].values  # numpy格式
print('分类种类:', np.unique(y))

###################   数据集切分          ###################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# 特征标准化
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)   # 用训练集fit
X_test = scaler_X.transform(X_test)         # 测试集用训练集参数


# 逻辑回归
log_model = LogisticRegression()
'''penalty：使用指定正则化项，可以指定为’l1’或者’l2’，L1正则化可以抵抗共线性，还会起到特征选择的作用，不重要的特征系数将会变为0；L2正则化一般不会将系数变为0，但会将不重要的特征系数变的很小，起到避免过拟合的作用。
   C：正则化强度取反，值越小正则化强度越大
   n_jobs: 指定线程数
   random_state：随机数生成器'''
# 随机森林
RFC_model = RandomForestClassifier()
'''n_estimators：森林中树的数量，默认是10棵，如果资源足够可以多设置一些。
   max_features：寻找最优分隔的最大特征数，默认是"auto"。
   max_ depth：树的最大深度。
   min_ samples_split：树中一个节点所需要用来分裂的最少样本数，默认是2。
   min_ samples_leaf：树中每个叶子节点所需要的最少的样本数。'''
# SVM
svm_model = SVC(kernel='linear', C=1.0, random_state=0)
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
# 梯度提升树
gbc_model = GradientBoostingClassifier()
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
# 极端梯度增强随机森林
lce_model = LCEClassifier(
    # ================= 外层随机森林参数 =================
    n_estimators=10,          # 外层随机森林的基学习器数量（相当于有30棵“大树”，每棵树其实是一个XGBoost）
    bootstrap=True,           # 是否采用Bootstrap抽样（有放回采样）
    max_samples=0.8,          # 每个基学习器训练时采样的样本
    max_features="sqrt",      # 每个基学习器随机使用 sqrt(特征数) 个特征
    max_depth=3,              # 外层弱学习器（类似树）的最大深度，防止过拟合
    min_samples_leaf=1,       # 外层树的叶子最小样本数
    metric="accuracy",        # 评估指标，可以改为 "f1" 或 "auc"
    n_iter=1,                 # 内层XGBoost超参数搜索的迭代次数（尝试多少组超参数）
    # ================= 内层基学习器（XGBoost）参数 =================
    base_learner="xgboost",          # 指定基学习器为XGBoost
    base_n_estimators=(100,),        # XGBoost的树数量候选值
    base_max_depth=(3,),             # XGBoost的树深度候选值
    base_learning_rate=(1,),         # 学习率候选值，越小越稳定但训练更慢
    base_gamma=(0,),                 # 节点分裂的最小损失下降，越大越保守
    base_min_child_weight=(1,),      # 子节点所需的最小样本权重，越大越保守
    base_subsample=(1.0,),           # 训练时的样本采样比例
    base_colsample_bytree=(1.0,),    # 每棵树训练时的特征采样比例
    base_reg_alpha=(0,),             # L1正则化系数（稀疏性）无
    base_reg_lambda=(0,),            # L2正则化系数（防止过拟合）
    base_booster=("gbtree",),        # 提升器类型（gbtree=树模型，常用）
    # ================= 运行参数 =================
    n_jobs=-1,                # 使用所有CPU核心并行
    random_state=42,          # 固定随机种子，保证结果可复现
    verbose=0                 # 日志输出等级（0=不输出，1=输出进度）
)
def evaluate_model(X,y,w='',model=None,X_test=None):
    if w == "训练":
        n_folds = 3
        model_names = ['log', 'svm', 'RFC', 'gbc','lce']
        model_dic = [log_model,svm_model,RFC_model,gbc_model,lce_model]
        # 交叉验证结果
        cv_score_list = []
        # 各个分类模型预测的y值列表
        pre_y_list = []
        for model in tqdm(model_dic):
            # 将每个分类模型导入交叉检验
            scores = cross_val_score(model, X, y, cv=n_folds,scoring='accuracy', error_score='raise')
            # 将交叉检验结果存入结果列表
            cv_score_list.append(scores)
            # 将分类训练中得到的预测y存入列表
            pre_y_list.append(cross_val_predict(model, X, y, cv=n_folds))
        ### 模型效果指标评估 ###
        # 分类评估指标列表
        model_metrics_list = []
        # 循环每个模型的预测结果
        for pre_y in pre_y_list:
            # 临时结果列表
            tmp_list = []
            # 计算每个分类指标结果
            tmp_score = accuracy_score(y, pre_y) # 正确的样本占总样本的比例
            tmp_list.append(tmp_score)
            # 每类先算指标，再平均
            tmp_score1 = recall_score(y, pre_y,average='macro')  # 实际为正的样本中，被正确预测为正的比例
            tmp_list.append(tmp_score1)
            tmp_score2 = precision_score(y, pre_y,average='macro') # 预测为正的样本中，真正为正的比例
            tmp_list.append(tmp_score2)
            tmp_score3 = f1_score(y, pre_y,average='macro') # 精确率和召回率的调和平均
            tmp_list.append(tmp_score3)
            # 将结果存入分类评估列表
            model_metrics_list.append(tmp_list)
        df_score = pd.DataFrame(cv_score_list, index=model_names)
        # 计算每行的均值和标准差
        df_score['均值'] = df_score.mean(axis=1)
        df_score['方差'] = df_score.std(axis=1)
        df_met = pd.DataFrame(model_metrics_list, index=model_names, columns=['准确率','召回率','精确率','F1分数'])
        print('-----------------------------训练-----------------------------')
        # 各个交叉验证的结果，准确率
        print(df_score)
        # 各种评估结果
        print(df_met)
    if w == "稳定":
        n_folds = 3  # 交叉验证折数
        # 交叉验证评分,内部已经拟合了模型
        scores = cross_val_score(model, X, y, cv=n_folds, error_score='raise')   # 该模型与fit无关
        # 预测结果确实对应 X 的每个样本，但每个样本都是在它没参与训练的模型上预测的。
        y_pred = cross_val_predict(model, X, y, cv=n_folds)
        # 分类评估指标
        metrics_list = [
            accuracy_score(y, y_pred),
            recall_score(y, y_pred, average='micro'),
            precision_score(y, y_pred, average='micro'),
            f1_score(y, y_pred, average='micro')
        ]
        # 构建 DataFrame 输出
        df_score = pd.DataFrame([scores])
        df_metrics = pd.DataFrame([metrics_list],columns=['准确率', '召回率', '精确率', 'F1分数'])
        print('-----------------------------训练-----------------------------')
        print('交叉验证结果:')
        print(df_score)
        print(f"Mean CV accuracy: {scores.mean():.4f}")
        print(f"Std CV accuracy: {scores.std():.4f}")
        print('性能评估指标:')
        print(df_metrics)
    if w == "预测":
        y_pred = model.fit(X, y).predict(X_test)
        print('-----------------------------预测-----------------------------')
        print('模型预测结果:', [round(i,4) for i in y_pred])
# 敏感性分析
def mingan():
    import shap
    lce_model.fit(X_train,y_train)
    explainer = shap.KernelExplainer(lce_model.predict_proba,shap.sample(X_train, 10))
    shap_values = explainer.shap_values(X_test[:5])   # (样本量，特征数，类别数)
    #print(shap_values)
    # 计算全局平均SHAP绝对值
    mean_abs_shap = np.abs(shap_values).mean(axis=(0,2))
    print('每个特征的总体平均贡献占比：\n',mean_abs_shap/sum(mean_abs_shap))
    mean_abs_shap1 = np.abs(shap_values).mean(axis=0).T
    print('每个类别的不同特征贡献占比：\n',mean_abs_shap1/mean_abs_shap1.sum(axis=1, keepdims=True))
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
    evaluate_model(X_train_noisy,y_train,w='稳定',model=lce_model)
    plt.ioff()

# 不确定性检验
def buqueding():
    y_pred = lce_model.fit(X_train,y_train).predict(X_train)
    #print('-----------------------------预测-----------------------------')
    #print('预测结果:', y_pred)
    # 找到预测错误的索引
    wrong_idx = np.where(y_pred != y_train)[0]
    print("错误样本数:", len(wrong_idx))
    probs = lce_model.predict_proba(X_train)
    #print("错误样本预测概率:", probs[wrong_idx])
    #print("错误样本预测标签:", y_train[wrong_idx])
    #print('预测详细概率：\n',probs)  # 预测概率
    confidence = np.max(probs, axis=1)  # 最大类别概率
    print('预测平均概率：\n',confidence.mean())
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)  # 熵，越低越确定
    K = probs.shape[1]
    print('归一化熵值：\n',entropy / np.log(K))
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:,-1] - sorted_probs[:,-2]  # 分类边际，越大越确定
    print('边际：\n',margin)
    # 保存文件
    df_wrong = pd.DataFrame(probs[wrong_idx], columns=np.unique(y))
    df_wrong = df_wrong.round(4)
    df_wrong["正确类别"] = y_train[wrong_idx]
    df_wrong["错误预测概率"] = confidence[wrong_idx].round(4)
    # 建立类别值 -> 列索引的映射
    class_to_col = {cls: i for i, cls in enumerate(df_wrong.columns)}
    # 把 "正确类别" 转换为列索引
    col_idx = df_wrong["正确类别"].map(class_to_col).values
    df_wrong["正确类别概率"] = probs[wrong_idx,col_idx].round(4)
    df_wrong.to_csv("错误预测样本.csv", index=False)

# 训练
evaluate_model(X_train,y_train,w='训练')
print('---------------------敏感性分析---------------------')
mingan()
print('---------------------稳定性分析---------------------')
wending()
print('---------------------不确定性分析---------------------')
buqueding()