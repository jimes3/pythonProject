import pandas as pd
import matplotlib.pyplot as plt
# 操作1
frame = pd.read_csv('股票数据.csv',encoding='GBK')
frame = frame.set_index('日期')
frame.index = pd.to_datetime(frame.index)
print(frame)

print(frame['收盘价'].shift(-1,freq='D')-frame['收盘价'])

# 操作2
plt.figure(figsize=(8,4),dpi=100)
plt.plot(frame['收盘价'])
plt.show()

# 平滑处理
plt.figure(figsize=(8,4),dpi=100)
plt.plot(frame['收盘价'].rolling(200).mean())
plt.show()

# 操作3
plt.figure(figsize=(8,4),dpi=100)
plt.plot(frame['收盘价'].pct_change())
plt.show()

def f(x):
    return x*x
plt.figure(figsize=(8,4),dpi=100)
plt.plot(frame['收盘价'].rolling(10).mean().apply(f))
plt.show()
