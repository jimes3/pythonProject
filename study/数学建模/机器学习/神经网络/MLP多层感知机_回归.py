import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\Jimes\PycharmProjects\pythonProject\study\数学建模\机器学习\神经网络\TCN\ID,crim,zn,indus,chas,nox,rm,age,di.csv",
                 usecols=['lstat','rm','crim','age','indus'])
# 自变量
train_x = df[['rm','crim','age','indus']].values
train_x = MinMaxScaler().fit_transform(train_x)    #标准化
# 因变量
train_y = df['lstat'].values
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
    nn.Linear(4, 16),
    nn.ReLU(),                   # 输入层激活
    #ResidualBlock(3, 3, dropout=0),  # 输入输出相同
    #ResidualBlock(32, 64, dropout=0),  # 输入输出不同，会自动加 projection
    nn.Linear(16, 1)# 输出层一般不用激活（回归），分类时用 sigmoid/softmax
)
# 指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# 初始化网络、损失函数和优化器
net = net.to(device)
criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()   # 多分类使用，最后几个节点有几类，内部自带 Softmax
#criterion = nn.BCEWithLogitsLoss()   # 二分类使用，内部自带 Sigmoid 转概率
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# 训练
epochs = 1200
losses = []
for epoch in range(epochs):
    net.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output.squeeze(), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 100 == 0:
        losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
# 画 loss 曲线
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
# 测试
pred = net(train_x.to(device)).detach().cpu().numpy()
#print("预测值：")
#print(pred)
# 画出原始值的曲线
plt.plot(range(len(train_y)), train_y, color='k', label='y')
# 画出各个模型的预测线
plt.plot(range(len(train_y)), pred.ravel(), 'r', label='pred')
plt.title('MLP')
plt.legend(loc='upper left')
plt.show()