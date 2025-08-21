import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("C:\\Users\Jimes\PycharmProjects\pythonProject\study\数学建模\机器学习\神经网络\TCN\ID,crim,zn,indus,chas,nox,rm,age,di.csv",
                 usecols=['rad','rm','crim','age','indus'])
# 自变量
train_x = df[['rm','crim','age','indus']].values
train_x = MinMaxScaler().fit_transform(train_x)    #标准化
# 因变量
train_y = df['rad'].values
print('分类种类:', np.unique(train_y))
# 确保标签在 [0, num_classes-1]
# 转成 Tensor
train_x = torch.tensor(train_x, dtype=torch.float32)  # shape: [batch, 1, seq_len]
train_y = torch.tensor(train_y, dtype=torch.float32)     # shape: [batch]

# 创建 Dataset
dataset = TensorDataset(train_x, train_y)
# 创建 DataLoader
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # 投影层，输入输出维度不同时使用
        self.projection = None
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
    def forward(self, x):
        identity = x
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        if self.projection is not None:
            identity = self.projection(identity)
        out = out + identity
        return self.relu(out)

net = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    ResidualBlock(32, 64, dropout=0),  # 可选残差块
    nn.Linear(64, 25)           # 输出类别数
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

net = net.to(device)
criterion = nn.CrossEntropyLoss()  # 多分类
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# 训练网络
num_epochs = 1000
for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = net(inputs.to(device))
        # 计算损失
        loss = criterion(outputs, labels.to(device).long())
        # 反向传播
        loss.backward()
        # 更新权重
        optimizer.step()

        running_loss += loss.item()
    if epoch % 100 == 99:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
        running_loss = 0.0

# 测试网络
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in train_loader:
        outputs = net(inputs.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()

print(f'Accuracy of the network: {100 * correct / total:.2f}%')