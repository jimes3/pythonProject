import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文


df = pd.read_excel('地理表.xlsx',header = None)
df = np.array(df)

x0 = np.linspace(0, 0.02*(len(df[0]-1)),len(df[0]))
y0 = np.linspace(0, 0.02*(len(df[0])),len(df))
z0 = -df
x, y = np.meshgrid(x0, y0)

# 绘制结果
fig = plt.figure(figsize=(9, 6))
ax=plt.subplot(1, 2, 1,projection = '3d')
surf = ax.plot_surface(x, y, z0, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0.5, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f')
plt.title('地势图')
ax=plt.subplot(1, 2, 2,projection = '3d')
plt.contour(x, y, z0, levels=50)  # levels参数指定等深线的数量
plt.xlabel('X')
plt.ylabel('Y')
plt.title('等深线图')

plt.show()