import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------------- 1. 读取数据 ----------------
df = pd.read_csv("时间序列预测数据集.csv")
series = df['Temp'].values.astype(float)

# ---------------- 2. 数据归一化 ----------------
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1,1)).flatten()

# ---------------- 3. 定义时间序列 Dataset ----------------
class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len=5, pred_step=1):
        series = np.array(series, dtype=float)
        self.X, self.y = [], []
        for i in range(len(series) - seq_len - pred_step + 1):
            self.X.append(series[i:i+seq_len])
            self.y.append(series[i+seq_len+pred_step-1])
        self.X = np.array(self.X).reshape(-1, seq_len, 1)
        self.y = np.array(self.y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

seq_len = 5
pred_step = 1
dataset = TimeSeriesDataset(series_scaled, seq_len=seq_len, pred_step=pred_step)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# ---------------- 4. 定义 LSTM 模型 ----------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)       # out: (B, seq_len, hidden_size)
        out = out[:, -1, :]         # 取最后时间步
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRegressor(input_size=1, hidden_size=64, num_layers=1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---------------- 5. 训练 ----------------
epochs = 100
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")

# ---------------- 6. 预测 ----------------
model.eval()
with torch.no_grad():
    X_full = torch.tensor(dataset.X, dtype=torch.float32).to(device)
    y_pred = model(X_full).cpu().numpy()

# 反归一化
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_true_rescaled = scaler.inverse_transform(dataset.y)

# ---------------- 7. 可视化 ----------------
plt.figure(figsize=(12,5))
plt.plot(y_true_rescaled, label="True Temp")
plt.plot(y_pred_rescaled, label="Predicted Temp")
plt.xlabel("Day")
plt.ylabel("Temperature")
plt.title("LSTM Temperature Prediction")
plt.legend()
plt.show()
