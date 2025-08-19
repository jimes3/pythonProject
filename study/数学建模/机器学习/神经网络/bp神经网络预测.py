import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 数据
X = np.array([[1,2,5],[1,3,4],[1,6,2],[1,5,1],[1,8,4]], dtype=np.float32)
y = np.array([[2],[3],[3],[2],[3]], dtype=np.float32)

X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# 定义网络
class BPNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 3)  # 输入3，隐藏层3
        self.out = nn.Linear(3, 1)     # 输出1
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.hidden(x))  # 隐藏层使用tanh
        x = self.out(x)                # 输出层线性激活
        return x

# 初始化网络、损失函数和优化器
net = BPNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# 训练
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = net(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.5f}')

# 测试
pred = net(X_tensor).detach().numpy()
print("预测值：")
print(pred)
