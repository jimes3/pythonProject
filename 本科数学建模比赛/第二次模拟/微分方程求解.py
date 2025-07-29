import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文

def f(t, y):
    # 定义微分方程组
    dy1_dt = y[2] #u_t
    dy2_dt = y[3] #w_t
    dy3_dt = (6250*np.cos(t/1.4005) - 656.3616*y[2] - 1025*9.8*np.pi*y[0]
            +10000*(y[3]-y[2]) + 80000*(y[1]-y[0]))/(4866+1335.535)
    dy4_dt = -(10000*(y[3]-y[2]) + 80000*(y[1]-y[0]))/2433
    return [dy1_dt, dy2_dt, dy3_dt, dy4_dt]
def runge_kutta4(t0, y0, h, num):
    t = t0
    y = y0
    result = []
    for _ in range(int(num)):
        k1 = [h * val for val in f(t, y)]
        k2 = [h * val for val in f(t + h/2, [y[i] + k1[i]/2 for i in range(len(y))])]
        k3 = [h * val for val in f(t + h/2, [y[i] + k2[i]/2 for i in range(len(y))])]
        k4 = [h * val for val in f(t + h, [y[i] + k3[i] for i in range(len(y))])]
        y = [y[i] + (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6 for i in range(len(y))]
        t += h
        result.append(y)
    return result
# 示例用法
t0 = 0  # 初始时间
t1 = 100  # 总时间
y0 = [0,0,0,0]  # 初始条件
h = 0.001  # 步长
x = np.arange(t0,t1,h)
num = (t1-t0)/h  # 迭代次数
result = runge_kutta4(t0, y0, h, num)
result = np.array(result).T

fig,axes = plt.subplots(1,2,figsize=(9,6))

plt.subplot(1,2,1)
plt.plot(x,result[0],label='浮子',linewidth=1.5,c = 'red')
plt.plot(x,result[1],label='振子',linewidth=1.0)
plt.xlim(0,100)
plt.xlabel('时间 s')
plt.ylabel('位移 m')
plt.legend()
plt.subplot(1,2,2)
plt.plot(x,result[2],label='浮子',linewidth=1.5,c = 'red')
plt.plot(x,result[3],label='振子',linewidth=1.0)
plt.xlim(0,100)
plt.xlabel('时间 s')
plt.ylabel('速度 m/s')
plt.legend()
plt.show()

plt.plot(x,result[1]-result[0])
plt.show()