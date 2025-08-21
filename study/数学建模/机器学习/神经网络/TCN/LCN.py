import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
import warnings
warnings.filterwarnings("ignore")

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        """
        input_size: 输入特征数（多因素个数）
        output_size: 输出特征数（单因素=1，多步预测=步数）
        num_channels: 每层通道数列表，例如 [16, 32, 64]
        kernel_size: 卷积核大小
        dropout: dropout比例
        return_seq: 是否返回整个序列预测 (True=多步预测，False=单步预测)
        """
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.output_size = output_size
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        x: (batch, input_size, seq_len)
        输出:
          - return_seq=False: (batch, output_size)  单步预测
          - return_seq=True:  (batch, seq_len, output_size) 多步预测
        """
        y1 = self.tcn(x)                         # (batch, channels, seq_len)
        out = self.linear(y1.transpose(1, 2))    # (batch, seq_len, output_size)
        return out[:, -self.output_size, :]                 # 取最后一步，单步预测

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
print(torch.__version__)        # 看 PyTorch 版本
print(torch.cuda.is_available()) # 检查是否支持 CUD
# 指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# 多因素 → 单因素单步预测
model = TCN(input_size=1, output_size=1, num_channels=[16, 32, 64], kernel_size=3, dropout=0.05).to(device)
df = pd.read_csv("ID,crim,zn,indus,chas,nox,rm,age,di.csv",
                 usecols=['lstat','rm','crim','age','indus'])
# 自变量
train_x = df[['rm','crim','age','indus']].values
train_x = MinMaxScaler().fit_transform(train_x)    #标准化
train_x = train_x.reshape(train_x.shape[0],1,train_x.shape[1])
# 因变量
train_y = df['lstat'].values
# 转成 Tensor
train_x = torch.tensor(train_x, dtype=torch.float32)  # shape: [batch, 1, seq_len]
train_y = torch.tensor(train_y, dtype=torch.float32)     # shape: [batch]

# 创建 Dataset
dataset = TensorDataset(train_x, train_y)
# 创建 DataLoader
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset, batch_size=64, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 40
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        # x: (batch, 1, seq_len)，y: (batch,)
        optimizer.zero_grad()
        output = model(x)           # (batch, 1)
        loss = criterion(output.squeeze(), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
# 测试
pred = model(train_x.to(device)).detach().cpu().numpy()
print("预测值：")
print(pred)