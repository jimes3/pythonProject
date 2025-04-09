# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def func(x, y):
    return (x+y)*np.exp(-5.0*(x**2 + y**2))

x = np.linspace(-1, 1, 4)
y = np.linspace(-1,1,4)

x, y = np.meshgrid(x, y)
fvals = np.array([[1,2,3,4],
                  [4,5,6,7],
                  [7,8,9,10],
                  [10,11,12,13]])

fig = plt.figure(figsize=(9, 6))

ax=plt.subplot(1, 2, 1,projection = '3d')
surf = ax.plot_surface(x, y, fvals, rstride=2, cstride=2, cmap=cm.coolwarm,linewidth=0.5, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.colorbar(surf, shrink=0.5, aspect=5)#标注

#二维插值
newfunc = interpolate.interp2d(x, y, fvals, kind='cubic')#newfunc为一个函数

# 计算100*100的网格上的插值
xnew = np.linspace(-1,1,100)#x
ynew = np.linspace(-1,1,100)#y
fnew = newfunc(xnew, ynew)#仅仅是y值   100*100的值  np.shape(f_new) is 100*100
xnew, ynew = np.meshgrid(xnew, ynew)
ax2=plt.subplot(1, 2, 2,projection = '3d')
surf2 = ax2.plot_surface(xnew, ynew, fnew, rstride=2, cstride=2, cmap=cm.coolwarm,linewidth=0.5, antialiased=True)
ax2.set_xlabel('x_new')
ax2.set_ylabel('y_new')
ax2.set_zlabel('f_new(x, y)')
plt.colorbar(surf2, shrink=0.5, aspect=5)#标注

plt.show()