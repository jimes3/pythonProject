import torch
import torch.nn as nn          # 神经网络模块
import torch.nn.functional as F # 激活函数等
import torch.optim as optim     # 优化器
from torchvision import datasets, transforms # 数据集和预处理

# 设置随机种子保证可重复性
torch.manual_seed(42)

# 1. 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),         # 将PIL图像转为Tensor (0-1范围)
    transforms.Normalize((0.1307,), (0.3081,)) # 标准化：(input - mean)/std
])

train_dataset = datasets.MNIST(
    './data',                      # 数据存储路径
    train=True,                    # 训练集
    download=False,  # 关闭下载
    transform=transform            # 应用预处理
)

test_dataset = datasets.MNIST(
    './data',
    train=False,                   # 测试集
    download=False,  # 关闭下载
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,  # 每批加载64个样本
    shuffle=True    # 打乱数据顺序
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1000 # 测试时使用更大的批次
)

# 2. 定义神经网络模型
class Net(nn.Module):  #继承PyTorch中所有神经网络模块的基类
    def __init__(self):
        super(Net, self).__init__()  # 初始化父类 nn.Module
        '''
        1. 初始化一个内部字典 _modules，用于存储子模块（如 nn.Linear、nn.Conv2d 等）。
        2. 初始化一个内部字典 _parameters，用于存储模型的可学习参数（如权重和偏置）。
        3. 初始化其他内部状态，例如钩子函数、设备信息等。
        如果你不调用 super().__init__()，这些基础设施将不会被初始化，导致模型无法正常工作。
        '''
        self.fc1 = nn.Linear(784, 128)  # 输入层 784 (28x28), 隐藏层 128
        self.fc2 = nn.Linear(128, 64)   # 隐藏层 64
        self.fc3 = nn.Linear(64, 10)    # 输出层 10 (10个数字类别)

    def forward(self, x):
        x = x.view(-1, 784)            # 展平输入图像
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                # 最后一层不使用激活函数（配合CrossEntropyLoss）
        return F.log_softmax(x, dim=1)  # 使用log_softmax输出概率

# 3. 初始化模型、损失函数和优化器
model = Net()
criterion = nn.CrossEntropyLoss()       # 自动包含softmax
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练过程
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()          # 梯度清零
        output = model(data)           # 前向传播
        loss = criterion(output, target)
        loss.backward()                # 反向传播
        optimizer.step()               # 参数更新

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 5. 测试过程
def test():
    model.eval() # 切换到评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 关闭梯度计算
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# 6. 运行训练和测试
for epoch in range(1, 6):  # 训练5个epoch
    train(epoch)
    test()

# 7. 保存模型
torch.save(model.state_dict(), "mnist_model.pth")

# 加载保存的模型
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# 单个样本预测
with torch.no_grad():
    sample = test_dataset[0][0].unsqueeze(0)
    output = model(sample)
    prediction = output.argmax(dim=1).item()
    print(f"Predicted digit: {prediction}")