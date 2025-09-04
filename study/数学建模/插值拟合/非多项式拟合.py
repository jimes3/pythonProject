import pylab
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.metrics import mean_squared_error # 均方误差

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

y = np.array([0.17464, 0.37198, 0.7709, 1.5964, 5.4856, 9.8399, 17.593, 28.455, 41.654, 18.216, 12.452, 5.8065, 4.8919, 2.7093, 1.8506, 1.9103, 1.911])
x = np.arange(1, len(y)+1, 1)

popt, pcov = curve_fit(func, x, y,maxfev=500000)            # 最小二乘法曲线拟合，popt为函数的参数list
y_pred = [func(i, popt[0], popt[1], popt[2]) for i in x]    # 直接用函数和函数参数list来进行y值的计算
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
import matplotlib.pyplot as plt
from scipy import stats
# 残差分析
residuals = y - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()
# 残差标准差
n, p = len(y), 3    # 预测数，参数数
sigma = np.sqrt(np.sum(residuals**2) / (n - p))
# 置信区间
alpha = 0.05
t_val = stats.t.ppf(1 - alpha/2, df=n - p)
ci_lower = y_pred - t_val * sigma
ci_upper = y_pred + t_val * sigma
# 可视化
plt.scatter(range(len(y)), y, label="Data")
plt.plot(range(len(y_pred)), y_pred, color="red", label="Fitted curve")
plt.fill_between(range(len(y)), ci_lower, ci_upper, color="pink", alpha=0.3, label="95% CI")
plt.legend()
plt.show()