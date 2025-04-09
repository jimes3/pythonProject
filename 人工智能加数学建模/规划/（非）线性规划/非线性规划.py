import numpy as np
from scipy.optimize import minimize,differential_evolution
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
# 定义目标函数
def objective(x):
    return 5 * x[0] - 2 * x[1] + x[2]
#定义约束条件
def constraint1(x):
    return x[0] + x[1] + x[2] - 1

def constraint2(x):
    return 1 - x[0] ** 2 - x[1] ** 2 - x[2] ** 2

def constraint3(x):
    return x[0] - x[2] ** 2

def constraint4(x):
    return x[0] + x[1] + x[2] - 1

# 定义初始点
x0 = np.array([0, 0, 0])

# 使用SLSQP算法求解非线性规划问题
solution = minimize(objective, x0, method='SLSQP',options={'maxiter':400}, constraints=[{'fun': constraint1, 'type': 'eq'},
                                                                                        {'fun': constraint2, 'type': 'ineq'},
                                                                                        {'fun': constraint3, 'type': 'ineq'},
                                                                                        {'fun': constraint4, 'type': 'ineq'}])
#fun：该参数就是costFunction你要去最小化的损失函数，将costFunction的名字传给fun
#x0: 猜测的初始值
#args=():优化的附加参数，默认从第二个开始
#method：该参数代表采用的方式，默认是BFGS, L-BFGS-B, SLSQP中的一种，可选TNC
#options：用来控制最大的迭代次数，以字典的形式来进行设置，例如：options={‘maxiter’:400}
#constraints: 约束条件，针对fun中为参数的部分进行约束限制,多个约束如下：
'''cons = ({'type': 'ineq', 'fun': lambda x: x[0] - x1min},
             {'type': 'ineq', 'fun': lambda x: -x[0] + x1max},  ineq表示大于等于号
             {'type': 'ineq', 'fun': lambda x: x[1] - x2min},
             {'type': 'ineq', 'fun': lambda x: -x[1] + x2max})'''
#tol: 目标函数误差范围，控制迭代结束
#callback: 保留优化过程
print(solution)
#无约束条件
res_differential_evolution = differential_evolution(func=objective, bounds=[(-1000,1000),(-1000,1000),(-1000,1000)])
print('differential_evolution()的结果为：\n', res_differential_evolution)