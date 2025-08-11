import numpy as np
from scipy import interpolate
import pylab as pl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出

###################################         一维插值         #####################################
y=[0,1,2,3,7,5,6,5,11,8,10]
x=np.linspace(0,10,len(y))
x_new=np.linspace(0,10,10*len(y))

pl.plot(x,y,"ro")
for kind in ["nearest","zero","slinear","quadratic","cubic"]:#插值方式
    #"nearest","zero"为阶梯插值
    #slinear 线性插值（默认）
    #"quadratic","cubic" 为2阶、3阶B样条曲线插值
    f=interpolate.interp1d(x,y,kind=kind)
    y_new=f(x_new)
    pl.plot(x_new,y_new,label=str(kind))
    print(f'{kind}:',y_new)
pl.legend(loc="lower right")
pl.show()
pl.close()

###################################         二维插值(已知函数)         #####################################
def func(x, y):
    return (x+y)*np.exp(-5.0*(x**2 + y**2))

x0 = np.linspace(-1, 1, 10)
y0 = np.linspace(-1,1,10)

x, y = np.meshgrid(x0, y0)
f_values = func(x,y)

fig = plt.figure(figsize=(9, 6))
#未插值的二维3D图
ax=plt.subplot(1, 2, 1,projection = '3d')
surf = ax.plot_surface(x, y, f_values, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0.5, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f')
#plt.colorbar(surf, shrink=0.4, aspect=5)        #标注颜色对应的值的大小

#二维插值
new_func = interpolate.interp2d(x, y, f_values, kind='cubic')      #newfunc为一个函数，实际上为一个拟合函数
x_new0 = np.linspace(-1, 1, 100)
y_new0 = np.linspace(-1, 1, 100)
f_new = new_func(x_new0, y_new0)  #得到的是在生成好的网格上的函数值
x_new, y_new = np.meshgrid(x_new0, y_new0)


#插值的二维3D图
ax2=plt.subplot(1, 2, 2,projection = '3d')
surf2 = ax2.plot_surface(x_new, y_new, f_new, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.1, antialiased=True)
ax2.set_xlabel('x_new')
ax2.set_ylabel('y_new')
ax2.set_zlabel('f_new')
#plt.colorbar(surf2, shrink=0.4, aspect=5)       #标注颜色对应的值的大小

plt.show()
plt.close()
###################################         二维插值(未知函数)         #####################################
# 已知数据点的坐标和函数值
x0 = np.array([0, 1, 2, 3, 4])
y0 = np.array([0, 1, 2, 4, 6])
f_values = np.array([1, 2, 3, 4, 5])

# 创建新的坐标网格
x_new0 = np.linspace(0, 4, 10)
y_new0 = np.linspace(0, 6, 10)

# 进行二维插值
f_new = griddata((x0, y0), f_values, (x_new0, y_new0), method='cubic')
print(f_new)
