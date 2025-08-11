import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文
def radial_basis_function_interpolation(x, y, x_interps, sigma):
    ####################################         计算径向基函数权重系数        ########################################
    n = len(x)     #获取插值长度
    A = np.zeros((n, n))      #生成与插值矩阵格式相同的零矩阵，以便后续存储插值矩阵
    for i in range(n):      #循环
        for j in range(n):     #迭代
            A[i, j] = np.exp(-0.5 * ((x[i] - x[j]) / sigma) ** 2) #以高斯径向基函数为核函数，计算插值矩阵
    w = np.dot(np.linalg.inv(A),y.T)  #计算得出权重系数w

    #################################       对待插值点进行插值计算        #########################################
    n_new = len(x_interps)  #获取代插值点的长度
    y_interps = np.zeros(n_new)   #生成初始待插值点用于存储插值的值
    for i in range(n_new):   #循环
        for j in range(n):   #迭代
            y_interps[i] += w[j] * np.exp(-0.5 * ((x_interps[i] - x[j]) / sigma) ** 2)   #通过高斯基函数计算每一个自变量上的函数值
    return y_interps  #返回插值点的值

# 生成随机的一维数据
np.random.seed(67)
x = np.random.rand(10)  # 已知数据点的 x 值
y = np.random.rand(10)  # 已知数据点的 y 值

# 生成等间隔的待插值点
x_interps = np.linspace(0, 1, 1000)

# 进行径向基函数插值
y_interp = radial_basis_function_interpolation(x, y, x_interps, sigma=0.1)

# 绘制结果
plt.plot(x, y, 'ro', label='Original Data')
plt.plot(x_interps, y_interp, 'b-', label='Interpolated Data')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('径向基函数插值')
plt.show()

