from lce import LCEClassifier
import pandas as pd
from sklearn.model_selection import cross_val_score,cross_val_predict    # 交叉验证
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

df = pd.read_csv("ID,crim,zn,indus,chas,nox,rm,age,di.csv",
                 usecols=['lstat','rm', 'rad','age','chas'])
# 自变量
X = df[['lstat', 'rm', 'age']].values
# 因变量
y = df['rad'].values
print('分类种类:', np.unique(y))

###################   数据集切分          ###################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  # train_test_split方法分割数据集

###############          标准化         #####################
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)    #标准化
X_test = StandardScaler().fit_transform(X_test)    #标准化

# 初始化 LCEClassifier
lce_model = LCEClassifier(
    # ================= 外层随机森林参数 =================
    n_estimators=50,          # 外层随机森林的基学习器数量（相当于有30棵“大树”，每棵树其实是一个XGBoost）
    bootstrap=True,           # 是否采用Bootstrap抽样（有放回采样）
    max_samples=0.8,          # 每个基学习器训练时采样80%的样本
    max_features="sqrt",      # 每个基学习器随机使用 sqrt(特征数) 个特征
    max_depth=3,              # 外层弱学习器（类似树）的最大深度，防止过拟合
    min_samples_leaf=1,       # 外层树的叶子最小样本数
    metric="accuracy",        # 评估指标，可以改为 "f1" 或 "auc"
    n_iter=1,                # 内层XGBoost超参数搜索的迭代次数（尝试多少组超参数）

    # ================= 内层基学习器（XGBoost）参数 =================
    base_learner="xgboost",   # 指定基学习器为XGBoost
    base_n_estimators=(50, 100),    # XGBoost的树数量候选值
    base_max_depth=(3, 6),             # XGBoost的树深度候选值
    base_learning_rate=(0.1,),  # 学习率候选值，越小越稳定但训练更慢
    base_gamma=(0, 1),                 # 节点分裂的最小损失下降，越大越保守
    base_min_child_weight=(1, 5),     # 子节点所需的最小样本权重，越大越保守
    base_subsample=(0.7, 0.9, 1.0),       # 训练时的样本采样比例
    base_colsample_bytree=(0.7, 0.9, 1.0),# 每棵树训练时的特征采样比例
    base_reg_alpha=(0,),         # L1正则化系数（稀疏性）无
    base_reg_lambda=(0.1,),      # L2正则化系数（防止过拟合）
    base_booster=("gbtree",),             # 提升器类型（gbtree=树模型，常用）

    # ================= 运行参数 =================
    n_jobs=-1,                # 使用所有CPU核心并行
    random_state=42,          # 固定随机种子，保证结果可复现
    verbose=0                 # 日志输出等级（0=不输出，1=输出进度）
)

def evaluate_model(model, X, y, w='', X_test=0):
    if w == "预测":
        # 拟合并预测
        y_pred = model.fit(X, y).predict(X_test)
        print('-----------------------------预测-----------------------------')
        print('lce_model预测结果:', y_pred)
    if w == "训练":
        n_folds = 5  # 交叉验证折数
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
# 交叉验证训练
#evaluate_model(lce_model,X_train,y_train,w="训练")
# 预测测试集
#evaluate_model(lce_model,X_train,y_train,w="预测",X_test=X_test)

# 不确定性检验
y_pred = lce_model.fit(X_train,y_train).predict(X_train)
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

# 敏感性分析
import shap
explainer = shap.KernelExplainer(lce_model.predict_proba,shap.sample(X_train, 10))
shap_values = explainer.shap_values(X_test[:5])   # (样本量，特征数，类别数)
#print(shap_values)
# 计算全局平均SHAP绝对值
mean_abs_shap = np.abs(shap_values).mean(axis=(0,2))
print('每个特征的总体平均贡献：\n',mean_abs_shap)
mean_abs_shap1 = np.abs(shap_values).mean(axis=0).T
print('每个类别的不同特征贡献：\n',mean_abs_shap1)