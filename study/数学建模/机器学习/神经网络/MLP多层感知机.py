import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 数据
X = np.array([[1,2,5],[1,3,4],[1,6,2],[1,5,1],[1,8,4]], dtype=np.float32)
y = np.array([[2],[3],[3],[2],[3]], dtype=np.float32)

X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # 如果输入输出维度不同，用线性层做投影
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
            identity = self.projection(identity) # 投影层
        out = out + identity
        return self.relu(out)
# 定义 MLP
net = nn.Sequential(
    nn.Linear(3, 3),
    nn.ReLU(),                   # 输入层激活
    ResidualBlock(3, 3, dropout=0),  # 输入输出相同
    #ResidualBlock(32, 64, dropout=0),  # 输入输出不同，会自动加 projection
    nn.Linear(3, 1)# 输出层一般不用激活（回归），分类时用 sigmoid/softmax
)
# 指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# 初始化网络、损失函数和优化器
net = net.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)

# 训练
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = net(X_tensor.to(device))
    loss = criterion(output, y_tensor.to(device))
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.5f}')

# 测试
pred = net(X_tensor.to(device)).detach().cpu().numpy()
print("预测值：")
print(pred)
