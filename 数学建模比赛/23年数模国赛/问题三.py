import numpy as np
import matplotlib.pyplot as plt

D = 110+3704*np.tan(np.deg2rad(1.5)) # 最深的海水深度
loc = [] # 储存测线的横坐标
loc1 = [] # 储存w1的值
loc2 = [] # 储存w2的值
WW = [] # 储存已知的覆盖宽度

g3 = np.sqrt(3)
s28 = np.sin(np.deg2rad(28.5))
s31 = np.sin(np.deg2rad(31.5))
t1 = np.tan(np.deg2rad(1.5))
c1 = np.cos(np.deg2rad(1.5))

d = D*g3/(2*s28+t1*g3) # 最开始的测线位置
w1 = d
w2 = g3/(2*s31)*c1*(D-d*np.tan(np.deg2rad(1.5)))*np.cos(np.deg2rad(1.5))
W = w1+w2
d -= 0.1*W
WW.append(0.9*W) # 去除10%的边缘不精准
loc.append(round(d,2))
loc1.append(w1)
loc2.append(w2)
while True:
    d = (D*g3+1.8*sum(WW)*s28)/(2*s28+t1*g3)
    if d >7408:
        break
    w1 = g3/(2*s28)*c1*(D-d*np.tan(np.deg2rad(1.5)))
    w2 = g3/(2*s31)*c1*(D-d*np.tan(np.deg2rad(1.5)))
    W = w1+w2
    WW.append(W)
    loc.append(round(d,2))
    loc1.append(w1)
    loc2.append(w2)

plt.vlines(loc,0,3704)
for i in range(len(loc)):
    plt.axvspan(loc[i]-loc1[i], loc[i]+loc2[i],ymin=0.05, ymax=0.95, alpha=0.1, color='red')
plt.show()

if 7408-loc[-1]>loc2[-1]:
    loc.append(7804)

print(loc) # 测线距离集合
print(loc1) # w1
print(loc2) # w2
print(WW) # W