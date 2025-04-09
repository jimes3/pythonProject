import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文

def radial_basis_function_interpolation(x, y, z, x_interps, y_interps, sigma_x, sigma_y):
    ####################################         计算径向基函数权重系数        ########################################
    n = len(x)     #获取插值长度
    x0 = np.repeat(x,n)  #从[1,2]变成[1,1,2,2]
    y0 = np.tile(y,n)   #从[1,2]变成[1,2,1,2]
    A = np.zeros((n**2, n**2))      #生成与插值矩阵格式相同的零矩阵，以便后续存储插值矩阵
    z = z.ravel()  #使其变成一维的
    for i in range(n**2):      #循环
        for j in range(n**2):     #迭代
            A[i, j] = np.exp(-0.5 * (((x0[i] - x0[j]) / sigma_x) ** 2 + ((y0[i] - y0[j]) / sigma_y) ** 2)) #以高斯径向基函数为核函数，计算插值矩阵
    w = np.dot(np.linalg.inv(A),z.T)  #计算得出权重系数w
    #################################       对待插值点进行插值计算        #########################################
    n_new = len(x_interps)  #获取代插值点的长度
    f_interps = np.zeros((n_new,n_new))   #生成初始待插值点用于存储插值的值
    for i in range(n_new):   #循环
        for v in range(n_new):   #迭代
            for j in range(n**2):   #迭代每一个已知点的权重值
                f_interps[i,v] += w[j] * np.exp(-0.5 * (((x_interps[i] - x0[j]) / sigma_x) ** 2 + ((y_interps[v] - y0[j]) / sigma_y) ** 2))   #通过高斯基函数计算每一个自变量上的函数值
    return f_interps        #返回插值点的值

#初始值,这里是网格数据
x0 = np.linspace(0, 7, 5)
y0 = np.linspace(0, 7, 5)
z0 = np.array([[3.9, 3.6, 3.5, 3.2, 2.8],
               [2.7, 2.6, 2.4, 2.1, 1.9],
              [3,2,5,3,4],
              [3.9, 3.6, 3.5, 3.2, 2.8],
              [2.7, 2.6, 2.4, 2.1, 1.9]])
# 生成等间隔的待插值点
x_interps0 = np.linspace(0, 7, 50)
y_interps0 = np.linspace(0, 7, 50)

#网格点操作
x, y = np.meshgrid(x0, y0)
x_interps, y_interps = np.meshgrid(x_interps0, y_interps0)

###################################       进行径向基函数插值         #################################################
f_interps = radial_basis_function_interpolation(x0, y0, z0, x_interps0,y_interps0, sigma_x=0.6,sigma_y=0.6)
print(f_interps)
# 绘制结果
fig = plt.figure(figsize=(9, 6))
#未插值的二维3D图
ax=plt.subplot(1, 2, 1,projection = '3d')
surf = ax.plot_surface(x, y, z0, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0.5, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f')

#插值的二维3D图
ax2=plt.subplot(1, 2, 2,projection = '3d')
surf2 = ax2.plot_surface(x_interps, y_interps, f_interps, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.1, antialiased=True)
ax2.set_xlabel('x_new')
ax2.set_ylabel('y_new')
ax2.set_zlabel('f_new')

plt.show()