'''一阶微分方程，且有n个变量的灰色模型'''
import numpy as np
import math as mt
import matplotlib.pyplot as plt

# 1.这里我们将 a 作为我们的特征序列 x0,x1,x2,x3作为我们的相关因素序列
a = [560823, 542386, 604834, 591248, 583031, 640636, 575688, 689637, 570790, 519574, 614677]
x0 = [104, 101.8, 105.8, 111.5, 115.97, 120.03, 113.3, 116.4, 105.1, 83.4, 73.3]
x1 = [135.6, 140.2, 140.1, 146.9, 144, 143, 133.3, 135.7, 125.8, 98.5, 99.8]
x2 = [131.6, 135.5, 142.6, 143.2, 142.2, 138.4, 138.4, 135, 122.5, 87.2, 96.5]
x3 = [54.2, 54.9, 54.8, 56.3, 54.5, 54.6, 54.9, 54.8, 49.3, 41.5, 48.9]

# 2.我们对其进行一次累加
def AGO(m):
    m_ago = [m[0]]
    add = m[0] + m[1]
    m_ago.append(add)
    i = 2
    while i < len(m):
        # print("a[",i,"]",a[i])
        add = add + m[i]
        # print("->",add)
        m_ago.append(add)
        i += 1
    return m_ago

a_ago = AGO(a)
x0_ago = AGO(x0)

x1_ago = AGO(x1)
x2_ago = AGO(x2)
x3_ago = AGO(x3)

xi = np.array([x0_ago, x1_ago, x2_ago, x3_ago])
print("xi", xi)

# 3.紧邻均值生成序列
def JingLing(m):
    Z = []
    j = 1
    while j < len(m):
        num = (m[j] + m[j - 1]) / 2
        Z.append(num)
        j = j + 1
    return Z

Z = JingLing(a_ago)
# print(Z)

# 4.求我们相关参数
Y = []
x_i = 0
while x_i < len(a) - 1:
    x_i += 1
    Y.append(a[x_i])
Y = np.mat(Y).T
Y.reshape(-1, 1)
print("Y.shape:", Y.shape)

B = []
b = 0
while b < len(Z):
    B.append(-Z[b])
    b += 1
B = np.mat(B)
B.reshape(-1, 1)
B = B.T
print("B.shape:", B.shape)
X = xi[:, 1:].T
print("X.shape:", X.shape)
B = np.hstack((B, X))
print("B-final:", B.shape)

# 可以求出我们的参数
theat = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
# print(theat)
al = theat[:1, :]
al = float(al)
# print("jhjhkjhjk",float(al))
b = theat[1:, :].T
print(b)
print("b.shape:", b.shape)
b = list(np.array(b).flatten())

# 6.生成我们的预测模型
U = []
k = 0
i = 0
# 计算驱动值
for k in range(11):
    sum1 = 0
    for i in range(4):
        sum1 += b[i] * xi[i][k]
        print("第", i, "行", "第", k, '列', xi[i][k])
        i += 1
    print(sum1)
    U.append(sum1)
    k += 1
print("驱动值:", U)

# 计算完整公式的值
F = []
F.append(a[0])

f = 1
while f < len(a):
    F.append((a[0] - U[f - 1] / al) / mt.exp(al * f) + U[f - 1] / al)
    f += 1
print("公式值：", F)

# 做差序列
G = []
G.append(a[0])
g = 1
while g < len(a):
    G.append(F[g] - F[g - 1])
    g += 1
print("预测值:", G)

r = range(11)
t = list(r)

plt.plot(t, a, color='r', linestyle="--", label='true')
plt.plot(t, G, color='b', linestyle="--", label="predict")
plt.legend(loc='upper right')

# 7.1绝对误差序列
'''import numpy as np
A=np.array(a)
G00=np.array(G)
abc=abs(A-G00)
abc简化'''
def abERR(m, n):
    err = []
    i = 0
    while i < len(m):
        num = abs(m[i] - n[i])
        err.append(num)
        i += 1
    return err
abErr = abERR(a, G)

# 7.1相对误差序列
def oppERR(m, n):
    err = []
    i = 0
    while i < len(m):
        num = m[i] / n[i]
        num1 = str(num * 100) + '%'
        err.append(num1)
        i += 1
    return err
oppErr = oppERR(abErr, a)

# 7.1 计算原始序列标准差S1
def MEAN1(m):
    add = 0
    i = 0
    while i < len(m):
        # print("a[",i,"]",a[i])
        add = add + m[i]
        # print("->",add)
        i += 1

    mean1 = add / len(m)
    return mean1
a_mean = MEAN1(a)

def Std1(m):
    Std1 = []
    j = 0
    add = 0
    while j < len(m):
        num = (m[j] - a_mean) ** 2
        add = add + num
        j = j + 1

    newlen = len(m) - 1
    std1 = add / newlen
    return std1
std11 = Std1(a)
print('原始序列标准差S1:',std11)

#7.2 计算小误差概率P
S0=0.6745*std11
print('小概率误差p:',S0)
# 7.3 计算绝对误差序列的标准差S2
import math
def MEAN2(m, n):
    add = 0
    i = 0
    while i < len(m):
        num = abs(m[i] - n[i])
        add = add + num
        i += 1
    mean2 = add / len(m)
    return mean2
G_mean = MEAN2(a, G)

def Std2(m):
    diff = []
    b = []
    j = 0
    add = 0
    while j < len(m):
        a = abs(m[j] - G_mean)
        diff.append(a)
        num = diff[j] ** 2
        add = add + num
        b.append(diff[j] - S0)  # 有结果得知，所有e1都小于S0，所以小误差概率P=1
        j = j + 1
    print("e1=", diff)
    print("b=", b)
    newlen = len(m) - 1
    std2 = add / newlen
    return std2
std22 = Std2(G)
print('绝对误差序列的标准差S2:',std22)

#7.4 计算方差比C
C=std11/std22
print('方差比C:',C)#C小于0.35，又由7.3知，小误差概率P=1，所以模型有好的预测精度
plt.show()