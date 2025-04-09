import pulp
import numpy as np

# 初始化
model = pulp.LpProblem("Multi-Objective_Programming", pulp.LpMinimize)

# 创建变量(假设模型有三个决策变量)
x1 = pulp.LpVariable("x1", 0, None, cat='Continuous')
x2 = pulp.LpVariable("x2", 0, None, cat='Continuous')
x3 = pulp.LpVariable("x3", 0, None, cat='Continuous')
x4 = pulp.LpVariable("x4", 0, None, cat='Binary')  #0-1变量
x5 = pulp.LpVariable("x5", 0, None, cat='Integer')
''' 'Continuous'：表示变量是连续的，即不限制变量取值的范围。
    'Integer'：表示变量是整数，即只能取整数值。
    'Binary'：表示变量是二进制，即只能取0或1两个值。
'''
# 定义目标函数 - 最小化z1和最小化z2
z1 = 2 * x1 + x2 + x3
z2 = x1 + 2 * x2 + 3 * x3
model += z1
model += z2

# 添加约束条件
model += x1 <= 5
model += x2 <= 8
model += x3 <= 10
model += x1 + x2 + x3 >= 7

# 求解模型
model.solve()

# 打印结果
print("Results:")
print("x1 = {}".format(x1.value()))
print("x2 = {}".format(x2.value()))
print("x3 = {}".format(x3.value()))
print("z1 = {}".format(z1.value()))
print("z2 = {}".format(z2.value()))