import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


df = np.array(pd.read_excel('地理表.xlsx',header = None))
def ju_mian(df):  # 拟合海底坡面
    X = np.linspace(0, 0.02*1852*(len(df[0]-1)),len(df[0]))
    Y = np.linspace(0, 0.02*1852*(len(df[0])),len(df))
    Z = -df
    X, Y = np.meshgrid(X, Y)
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    def nihe_plane_fit(xishuyouxiao, X, Y, Z):
        a, b, c, d = xishuyouxiao
        return a * X + b * Y + c * Z + d
    initial_guess = [1.1, 1.1, 1.1, 1.1]
    a, b, c, d = (least_squares(nihe_plane_fit, initial_guess, args=(X, Y, Z))).x
    return a,b,c,d
def yuansu(df): # 求解覆盖宽度
    a,b,c,d = ju_mian(df)  # 坡面方程的系数
    fa1 = np.array([a,b,c]) # 海底拟合坡面的法向量

    thetaa = np.array([a,b,0])
    co_si = (np.dot([1,0,0], thetaa) /  # 计算两个投影向量之间的夹角（弧度）
             (np.linalg.norm([0,0,1]) * np.linalg.norm(thetaa)))
    angle_r1 = np.arccos(co_si)
    angle_deg1 = 90-np.degrees(angle_r1)  # 坡面法向量投影于水平面与x轴的夹角

    xian = np.array([np.cos(np.deg2rad(angle_deg1+90)),
                     np.sin(np.deg2rad(angle_deg1+90)), 0])  # 测线的方向向量(投影)
    touying1 = (np.dot(xian, fa1) / np.linalg.norm(fa1)) * fa1 # 坡面上的投影
    co_si = (np.dot(touying1, xian) /  # 计算两个投影向量之间的夹角（弧度）
             (np.linalg.norm(touying1) * np.linalg.norm(xian)))
    angle_r = np.arccos(co_si)
    angle_deg2 = 90-np.degrees(angle_r)  # 将弧度转换为度数

    D = np.mean(df)  # 求得深度
    g3 = np.sqrt(3)
    s2 = np.sin(np.deg2rad(30-angle_deg2))
    s3 = np.sin(np.deg2rad(30+angle_deg2))
    c1 = np.cos(np.deg2rad(angle_deg2))
    w1 = g3/(2*s2)*c1*D  # 坡度较大则小于0
    w2 = g3/(2*s3)*c1*D
    w = w1+w2
    return w,w1,w2,angle_deg1

for i in range(1000):
    xy = [random.randint(0,200),random.randint(0,250)]
    df = np.array([row[max(0, xy[0] - 4) : min(len(df[0]), xy[0] + 4 + 1)]
                   for row in df[max(0, xy[1] - 4) : min(len(df), xy[1] + 4 + 1)]]) # 前x后y
    w,w1,w2,jiao = yuansu(df)
    print(w,w1,w2)
    print(jiao)
    x , y = xy[0],xy[1]
    angle = np.deg2rad(jiao)
    if np.abs(np.cos(angle)) < 1e-6:
        # 如果斜率不存在，绘制垂直于 x 轴的竖直线
        x_r = np.array([x, x])
        y_r = np.array([y - 4, y + 4])
    else:
        a = np.tan(angle)  # 直线斜率
        ii = np.cos(angle)
        x_r = np.linspace(x - 4 * ii, x + 4 * ii, 5)
        # 计算对应的y范围
        y_r = a * (x_r - x) + y
    df = np.array(pd.read_excel('地理表.xlsx',header = None))
    # 绘制直线或竖直线
    plt.plot(x_r, y_r)
    plt.scatter(x, y, color='red',s=0.5)
plt.grid(True)

# 显示图形
plt.show()

