# -*-coding:utf-8 -*-
import numpy as np
from scipy import interpolate
import pylab as pl
#x = eval(input('请输入x值：'))

y=[0,1,2,3,4,5,6,7,8,9,10]
x=np.linspace(0,10,len(y))
#y=np.sin(x)
#y = eval(input('请输入y值：'))
xnew=np.linspace(0,10,101)
pl.plot(x,y,"ro")

for kind in ["nearest","zero","slinear","quadratic","cubic"]:#插值方式
    #"nearest","zero"为阶梯插值
    #slinear 线性插值（默认）
    #"quadratic","cubic" 为2阶、3阶B样条曲线插值
    f=interpolate.interp1d(x,y,kind=kind)
    # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
    ynew=f(xnew)
    pl.plot(xnew,ynew,label=str(kind))
    print(f'{kind}:',ynew)
pl.legend(loc="lower right")
pl.show()