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

popt, pcov = curve_fit(func, x, y,maxfev=500000)                # 曲线拟合，popt为函数的参数list
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