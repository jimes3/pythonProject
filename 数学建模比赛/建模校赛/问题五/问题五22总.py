import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pylab
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

##########################     运输量数据      #################################
df = pd.read_excel('附件2(Attachment 2)2023-51MCM-Problem B.xlsx',header = 0)
df = df.set_index("日期(年/月/日) (Date Y/M/D)")

start_date = '2022-07-01'
end_date = '2022-09-30'

rows = df.loc[start_date:end_date]
dfz = rows.resample("B").sum()
dfz = np.array(dfz).ravel()
vn = np.sort(dfz)
vn = vn[vn>0]
#print(vn)
def normal_distribution(x, mu, sigma,C):
    """
    计算正态分布在x处的概率密度函数的值
    :param x: 自变量x
    :param mu: 均值
    :param sigma: 标准差
    :return: 在x处的概率密度函数的值
    """
    return C*norm.pdf(x, mu, sigma)
sc = [100000,0,[],[],[],[],[]]
for i in range(np.min(vn)+1):
    y = vn-i
    vn = vn[vn > 0]
    x = np.arange(0, len(vn), 1)
    popt, pcov = curve_fit(normal_distribution, x, y,maxfev=500000)
    y_pred = [normal_distribution(i, popt[0],popt[1],popt[2]) for i in x]
    #print('系数:',popt)
    #print('预测值:',y_pred)
    #mse = mean_squared_error(y, y_pred)
    me = abs(np.mean(y - y_pred))
    if me<sc[0]:
        sc[0]=me
        sc[1]=i
        sc[2]=x
        sc[3]=y
        sc[4]=y_pred
        sc[5]=popt
print(sc[1],sc[5][0],sc[5][1])
plot1 = pylab.plot(sc[2],sc[3], '*', label='original values')
plot2 = pylab.plot(sc[2],sc[4], 'r', label='fit values')
pylab.title('')
pylab.xlabel('')
pylab.ylabel('')
pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 1))
pylab.show()