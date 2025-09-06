import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# ---------------- 1. 读取数据 ----------------
df = pd.read_csv("时间序列预测数据集.csv")
series = df['Temp'].values.astype(float)

# ---------------- 2. 参数 ----------------
seq_len = 100     # 输入序列长度
n_steps = 5      # 预测未来 n 步
test_size = 0.2  # 测试集比例

# ---------------- 3. 数据归一化 ----------------
scaler = StandardScaler()
series_scaled = scaler.fit_transform(series.reshape(-1,1)).flatten()

# ---------------- 4. 构造多步训练样本 ----------------
X, y = [], []
for i in range(len(series_scaled) - seq_len - n_steps + 1):
    X.append(series_scaled[i:i+seq_len])
    y.append(series_scaled[i+seq_len:i+seq_len+n_steps])
X = np.array(X).reshape(-1, seq_len, 1)   # (samples, timesteps, features)
y = np.array(y)                            # (samples, n_steps)

# ---------------- 5. 划分训练/测试 ----------------
split_idx = int(len(X) * (1 - test_size))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ---------------- 6. 构建 LSTM ----------------
model = Sequential()
# 第一层 LSTM，返回序列以供下一层处理
model.add(LSTM(128, input_shape=(seq_len,1), return_sequences=True))
#model.add(Dropout(0.2))
# 第二层 LSTM，不返回序列
model.add(LSTM(64))
#model.add(Dropout(0.2))
# 输出 n 步
model.add(Dense(n_steps))
model.compile(optimizer='adam', loss='mse')

# ---------------- 7. 训练 ----------------
y = model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
# ---------------- 在训练集上预测 ----------------
y_train_pred_scaled = model.predict(X_train)
y_train_pred = scaler.inverse_transform(y_train_pred_scaled)
y_train_true = scaler.inverse_transform(y_train)

# ---------------- 计算训练集 MSE ----------------
from sklearn.metrics import mean_squared_error
mse_train = mean_squared_error(y_train_true, y_train_pred)
print("训练集 MSE:", mse_train)

# ---------------- 可视化 ----------------
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(y_train_true.flatten(), label='True Temp (Train)')
plt.plot(y_train_pred.flatten(), label='Predicted Temp (Train)')
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.title("LSTM Fit on Training Set")
plt.legend()
plt.show()

# ---------------- 8. 预测 ----------------
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)  # 反归一化
y_true = scaler.inverse_transform(y_test)
mse = mean_squared_error(y_true, y_pred)
print("测试集 MSE:", mse)
# ---------------- 9. 可视化 ----------------
plt.figure(figsize=(12,5))
plt.plot(y_true.flatten(), label='True Temp')
plt.plot(y_pred.flatten(), label='Predicted Temp')
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.title("Keras LSTM Multi-step Forecast")
plt.legend()
plt.show()
