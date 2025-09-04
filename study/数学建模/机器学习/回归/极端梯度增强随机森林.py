from lce import LCERegressor
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
import seaborn as sns
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文

# ---------------- 数据读取 ----------------
df = pd.read_csv("ID,crim,zn,indus,chas,nox,rm,age,di.csv",
                 usecols=['lstat','rm','crim','age','indus'])

# 自变量
X = df[['rm','crim','age','indus']].values
# 因变量
y = df['lstat'].values

# ---------------- 数据集切分 ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# ---------------- 标准化 ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- 初始化 LCERegressor ----------------
lce_model = LCERegressor(
    n_estimators=5,
    bootstrap=True,
    max_samples=0.8,
    max_features="sqrt",
    max_depth=3,
    min_samples_leaf=1,
    metric="neg_mean_squared_error",   # 回归指标
    n_iter=1,
    base_learner="xgboost",
    base_n_estimators=(50,),
    base_max_depth=(3, 6,),
    base_learning_rate=(0.1,),
    base_gamma=(0, 1, 5),
    base_min_child_weight=(1, 5,),
    base_subsample=(0.7, 0.9, 1.0),
    base_colsample_bytree=(0.7, 0.9, 1.0),
    base_reg_alpha=(0,),
    base_reg_lambda=(0.1, 1.0,),
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
    if w == "残差分析":   # 残差必须要近似正态
        y_pred = model.fit(X, y).predict(X)
        # 残差分析
        residuals = y - y_pred
        sns.histplot(residuals, kde=True)
        plt.title("残差正态分布检验")
        plt.show()
        import statsmodels.api as sm
        sm.qqplot(residuals, line='45', fit=True)
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
        print("Coverage:", coverage)
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
# ---------------- 使用 ----------------
# 交叉验证训练
evaluate_regressor(lce_model, X_train, y_train, w="训练")
# 用训练好的模型预测测试集
evaluate_regressor(lce_model, X_train, y_train, w="预测", X_test=X_test)
evaluate_regressor(lce_model, X_train, y_train, w="残差分析")

# 敏感性分析
import shap
explainer = shap.KernelExplainer(lce_model.predict,shap.sample(X_train, 100))
shap_values = explainer.shap_values(X_test[:3])

# 计算全局平均SHAP绝对值
mean_abs_shap = np.abs(shap_values).mean(axis=0)
print(mean_abs_shap)  # 每个特征的贡献