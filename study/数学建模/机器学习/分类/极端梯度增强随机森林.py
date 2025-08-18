from lce import LCEClassifier
import pandas as pd
from sklearn.model_selection import cross_val_score,cross_val_predict    # 交叉验证
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, average_precision_score, f1_score
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("ID,crim,zn,indus,chas,nox,rm,age,di.csv",
                 usecols=['lstat','rm', 'chas'])
# 自变量
X = df[['lstat', 'rm']].values
# 因变量
y = df['chas'].values
print('分类种类:', np.unique(y))

###################   数据集切分          ###################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  # train_test_split方法分割数据集

###############          标准化         #####################
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)    #标准化
X_test = StandardScaler().fit_transform(X_test)    #标准化

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

# 初始化 LCEClassifier
lce_model = LCEClassifier(
    # ================= 外层随机森林参数 =================
    n_estimators=30,          # 外层随机森林的基学习器数量（相当于有30棵“大树”，每棵树其实是一个XGBoost）
    bootstrap=True,           # 是否采用Bootstrap抽样（有放回采样）
    max_samples=0.8,          # 每个基学习器训练时采样80%的样本
    max_features="sqrt",      # 每个基学习器随机使用 sqrt(特征数) 个特征
    max_depth=3,              # 外层弱学习器（类似树）的最大深度，防止过拟合
    min_samples_leaf=1,       # 外层树的叶子最小样本数
    metric="accuracy",        # 评估指标，可以改为 "f1" 或 "auc"
    n_iter=20,                # 内层XGBoost超参数搜索的迭代次数（尝试多少组超参数）

    # ================= 内层基学习器（XGBoost）参数 =================
    base_learner="xgboost",   # 指定基学习器为XGBoost
    base_n_estimators=(100, 300, 500),    # XGBoost的树数量候选值
    base_max_depth=(3, 6, 9),             # XGBoost的树深度候选值
    base_learning_rate=(0.1,),  # 学习率候选值，越小越稳定但训练更慢
    base_gamma=(0, 1, 5),                 # 节点分裂的最小损失下降，越大越保守
    base_min_child_weight=(1, 5, 10),     # 子节点所需的最小样本权重，越大越保守
    base_subsample=(0.7, 0.9, 1.0),       # 训练时的样本采样比例
    base_colsample_bytree=(0.7, 0.9, 1.0),# 每棵树训练时的特征采样比例
    base_reg_alpha=(0,),         # L1正则化系数（稀疏性）无
    base_reg_lambda=(0.1, 1.0, 5.0),      # L2正则化系数（防止过拟合）
    base_booster=("gbtree",),             # 提升器类型（gbtree=树模型，常用）

    # ================= 运行参数 =================
    n_jobs=-1,                # 使用所有CPU核心并行
    random_state=42,          # 固定随机种子，保证结果可复现
    verbose=1                 # 日志输出等级（0=不输出，1=输出进度）
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
        scores = cross_val_score(model, X, y, cv=n_folds, error_score='raise')
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
evaluate_model(lce_model,X_train,y_train,w="训练")
# 预测测试集
evaluate_model(lce_model,X_train,y_train,w="预测",X_test=X_test)