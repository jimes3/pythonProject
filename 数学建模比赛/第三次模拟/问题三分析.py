import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文

y = [28582, 28190, 28595, 28180, 28606, 28172, 28609, 28165, 28619, 28161, 28617, 28159,
     28623, 28156, 28624, 28150, 28636, 28135, 28651, 28122, 28658, 28120, 28661, 28116]
x = [i for i in range(len(y))]
plt.plot(x,y,label='订购数量')
plt.plot(x,[28200 for _ in range(len(y))],label='要求数量')
plt.legend()
plt.show()
