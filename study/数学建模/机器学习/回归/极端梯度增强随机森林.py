from lce import LCERegressor
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ---------------- 数据读取 ----------------
df = pd.read_csv("ID,crim,zn,indus,chas,nox,rm,age,di.csv",
                 usecols=['lstat','rm','chas'])

# 自变量
X = df[['lstat','rm']].values
# 因变量
y = df['chas'].values

# ---------------- 数据集切分 ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# ---------------- 标准化 ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- 可视化 ----------------
sns.set(style='whitegrid', context='notebook')
sns.pairplot(df, height=2)
plt.savefig('数据关系.png', dpi=600)
plt.close()

corr = df.corr()
sns.heatmap(corr, cmap='GnBu_r', annot=True)
plt.savefig('相关度热力图.png', dpi=600)
plt.close()

# ---------------- 初始化 LCERegressor ----------------
lce_model = LCERegressor(
    n_estimators=30,
    bootstrap=True,
    max_samples=0.8,
    max_features="sqrt",
    max_depth=3,
    min_samples_leaf=1,
    metric="neg_mean_squared_error",   # 回归指标
    n_iter=20,
    base_learner="xgboost",
    base_n_estimators=(100, 300, 500),
    base_max_depth=(3, 6, 9),
    base_learning_rate=(0.1,),
    base_gamma=(0, 1, 5),
    base_min_child_weight=(1, 5, 10),
    base_subsample=(0.7, 0.9, 1.0),
    base_colsample_bytree=(0.7, 0.9, 1.0),
    base_reg_alpha=(0,),
    base_reg_lambda=(0.1, 1.0, 5.0),
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
        print('-----------------------------预测-----------------------------')
        print('LCERegressor预测结果:', [round(i,4) for i in y_pred])
    if w == "训练":
        n_folds = 5
        # 交叉验证评分 (neg_mean_squared_error)
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

# ---------------- 使用 ----------------
# 交叉验证训练
evaluate_regressor(lce_model, X_train, y_train, w="训练")
# 用训练好的模型预测测试集
evaluate_regressor(lce_model, X_train, y_train, w="预测", X_test=X_test)
