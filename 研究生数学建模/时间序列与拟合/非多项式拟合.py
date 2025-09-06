import pylab
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.metrics import mean_squared_error # 均方误差
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文
plt.style.use('ggplot')
def func(x, a, b, c):
    return a*x**2+b*x+c
def normal_distribution(x, mu, sigma,C):
    """
    计算正态分布在x处的概率密度函数的值
    :param x: 自变量x
    :param mu: 均值
    :param sigma: 标准差
    :return: 在x处的概率密度函数的值
    """
    return C*norm.pdf(x, mu, sigma)

y = np.random.normal(0, 1, 1000) + np.random.normal(loc=0, scale=0.001, size=1000)
x = np.arange(1, len(y)+1, 1)

popt, pcov = curve_fit(normal_distribution, x, y,maxfev=500000)            # 最小二乘法曲线拟合，popt为函数的参数list
y_pred = [normal_distribution(i, popt[0], popt[1], popt[2]) for i in x]    # 直接用函数和函数参数list来进行y值的计算
print('系数:',popt)
print('预测值:',y_pred)
print(mean_squared_error(y,y_pred))
plot1 = pylab.plot(x, y, '*', label='original values')
plot2 = pylab.plot(x, y_pred, 'r', label='fit values')
pylab.title('')
pylab.xlabel('')
pylab.ylabel('')
pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 1))
pylab.show()
#pylab.savefig('p1.png', dpi=200, bbox_inches='tight')

# 敏感性分析
def mingan():
    # 拟合后的参数
    mu, sigma, C = popt
    # 原始预测值
    y_pred = normal_distribution(x, mu, sigma, C)
    # 定义扰动敏感性函数
    def parameter_sensitivity(param_name, delta=0.01):
        param_map = {'mu': mu, 'sigma': sigma, 'C': C}
        sensitivities = []
        original_value = param_map[param_name]
        for factor in [1 - delta, 1 + delta]:  # ±delta
            param_map[param_name] = original_value * factor
            y_new = normal_distribution(x, param_map['mu'], param_map['sigma'], param_map['C'])
            mse_change = np.mean((y_pred - y_new)**2)  # 输出变化量
            sensitivities.append(mse_change)
        # 恢复原值
        param_map[param_name] = original_value
        return np.mean(sensitivities)
    # 计算每个参数的重要性
    param_importance = {p: parameter_sensitivity(p) for p in ['mu','sigma','C']}
    total = sum(param_importance.values())
    for p, val in param_importance.items():
        print(f"{p}的重要性占比: {val/total:.2%}")

# 稳定性分析
def wending():
    n_sim = 100
    popt_all = []
    mse_all = []
    for _ in range(n_sim):
        y_noisy = y + np.random.normal(0, 0.01, size=len(y))
        popt_noisy, _ = curve_fit(normal_distribution, x, y_noisy, maxfev=500000)
        popt_all.append(popt_noisy)
        # 用拟合参数计算预测值
        y_pred_noisy = normal_distribution(x, *popt_noisy)
        # 计算 MSE
        mse = mean_squared_error(y_noisy, y_pred_noisy)
        mse_all.append(mse)
    popt_all = np.array(popt_all)
    mse_all = np.array(mse_all)
    # 输出 MSE 均值和标准差
    print(f"MSE 均值: {np.mean(mse_all):.5f}")
    print(f"MSE 标准差: {np.std(mse_all):.5f}")
    import matplotlib.pyplot as plt
    labels = ['mu', 'sigma', 'C']
    plt.figure(figsize=(10,5))
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.hist(popt_all[:,i], bins=20, color='skyblue', edgecolor='k')
        plt.title(f'{labels[i]} 分布')
        plt.xlabel('参数值')
        plt.ylabel('频数')
    plt.tight_layout()
    plt.show()

# 不确定性
def buqueding():
    # 残差分析
    residuals = y - y_pred
    # 对 residuals 随机采样，否则时间太长
    sample_resid = np.random.choice(residuals.ravel(), size=len(y), replace=False)
    import seaborn as sns
    sns.histplot(sample_resid, kde=True)
    plt.title("残差正态分布检验")
    plt.show()
    import statsmodels.api as sm
    sm.qqplot(sample_resid, line='45', fit=True)
    plt.title("残差QQ图")  #靠近45度线表明符合正态
    plt.show()
    from scipy.stats import shapiro,normaltest, jarque_bera
    stat, p = normaltest(residuals)
    print('D’Agostino K² test p-value', round(p,5))
    stat, p = shapiro(residuals)
    print('Shapiro-Wilk test p-value:', round(p,5))
    stat, p = jarque_bera(residuals)
    print('Jarque-Bera test p-value:', round(p,5))
    if p > 0.05:
        print("残差近似正态")
    else:
        print("残差偏离正态")
    # 残差均值和标准差
    resid_mean = np.mean(residuals)
    resid_std = np.std(residuals)
    # 置信区间
    upper = y_pred + resid_mean + 1.96 * resid_std
    lower = y_pred + resid_mean - 1.96 * resid_std
    # 可视化
    plt.figure(figsize=(10,5))
    plt.plot(x, y, 'o', label='原始数据')
    plt.plot(x, y_pred, 'r', label='拟合均值')
    plt.fill_between(x, lower, upper, color='r', alpha=0.2, label='95%预测区间')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('残差驱动的不确定性分析（非蒙特卡洛版）')
    plt.legend()
    plt.show()

wending()