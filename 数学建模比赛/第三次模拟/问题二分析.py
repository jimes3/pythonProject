import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文

y = [28578,28197,28583,28194,28586,28190,28592,28184,28597,28178,28603,28171,
     28610,28160,28622,28153,28628,28146,28636,28142,28639,28137,28641,28135]
x = [i for i in range(len(y))]
plt.plot(x,y,label='订购数量')
plt.plot(x,[28200 for _ in range(len(y))],label='要求数量')
plt.legend()
plt.show()


plt.pie([3,4,4],labels=['A','B','C'],autopct= '%0.1f%%',textprops={'fontsize': 12} )
plt.show()

