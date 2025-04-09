import pylab
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error # 均方误差


def func(x, a, b, c):
    return a*x**2+b*x+c

df = pd.read_excel("D:\.jimes\下载\均值.xlsx",sheet_name='Sheet2',header=None)
df = np.array(df)

x = np.arange(1, 5, 1)
for i in range(len(df)):
    y = np.array(df[i])
    popt, pcov = curve_fit(func, x, y,maxfev=500000)                # 曲线拟合，popt为函数的参数list
    y_pred = [func(i, popt[0], popt[1], popt[2]) for i in x]    # 直接用函数和函数参数list来进行y值的计算
    print('系数:',popt)
    #print('预测值:',y_pred)
    print(f'mse:{mean_squared_error(y,y_pred):.4f}')
'''
plot1 = pylab.plot(x, y, '*', label='original values')
plot2 = pylab.plot(x, y_pred, 'r', label='fit values')
pylab.title('')
pylab.xlabel('')
pylab.ylabel('')
pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 1))
pylab.show()
'''