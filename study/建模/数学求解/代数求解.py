from scipy.optimize import fsolve
import numpy as np
from fractions import Fraction
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出

def func(x):  #方程式联解
    # 定义方程组
    eq1 = -0.8*x[0] + 0.6*x[1] +0.2*x[2]
    eq2 = 0.3*x[0] -x[1] +0.7*x[2]
    eq3 = 0.5*x[0]  -0.5*x[2]
    return [eq1, eq2,eq3]
# 求解方程组
result = fsolve(func, [1, 1,1])
print(result)

#求特征向量与特征值
matrix = np.array([[0.2, 0.6, 0.2],
                            [0.3, 0, 0.7],
                            [0.5, 0, 0.5]],dtype=float)
vector1 = np.matrix([[0.3, 0.4, 0.3]], dtype=float) #初始状态
for i in range(100):  #多次迭代后趋于稳定
    vector1 = vector1 * matrix
    print(f'迭代次数: {i+1}')
    print(vector1)

eigenvalue, eigenvector = np.linalg.eig(matrix)
# 转换特征值为分数
eigenvalues = [Fraction(eig_val).limit_denominator() for eig_val in np.real(eigenvalue)]
# 转换特征向量为分数
eigenvectors = [np.array([Fraction(val).limit_denominator() for val in eig_vector]) for eig_vector in np.real(eigenvector)]
print("特征值：", eigenvalue)
print("特征向量：", eigenvector) #竖着看