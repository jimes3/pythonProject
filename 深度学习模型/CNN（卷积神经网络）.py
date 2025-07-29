import torch
import torch.nn as nn          # 神经网络模块
import torch.nn.functional as F # 激活函数等
import torch.optim as optim     # 优化器
from torchvision import datasets, transforms # 数据集和预处理
print(torch.__version__)
print(torch.cuda.is_available())
# 设置随机种子保证可重复性
torch.manual_seed(42)
#哈哈哈
# 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),         # 将PIL图像转为Tensor (0-1范围)
    transforms.Normalize((0.5,), (0.5,)) # 标准化：(input - mean)/std,归一化到[-1, 1]
])

train_dataset = datasets.MNIST(
    './data',                 # 数据存储路径
    train=True,                    # 训练集
    download=False,                # 关闭下载
    transform=transform            # 应用预处理
)

test_dataset = datasets.MNIST(
    './data',
    train=False,                   # 测试集
    download=False,                # 关闭下载
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,  # 每批加载64个样本
    shuffle=False    # 打乱数据顺序
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1000,  # 测试时使用更大的批次
    shuffle=False    # 打乱数据顺序
)

# 定义CNN网络模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        '''
        in_channels:输入数据的通道数
        out_channels:输出数据的通道数,即卷积层中滤波器的数量
        kernel_size:卷积核（滤波器）的大小,3*3
        padding:输入数据的边缘填充的像素数.
        当 kernel_size=3 且 padding=1 时，输出特征图的高度和宽度与输入相同
        '''
        # 第二个卷积层
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        '''
        在每个 2x2 的区域内，取最大值作为输出。
        kernel_size:池化窗口的大小
        stride:池化窗口滑动的步长
        '''
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # MNIST图像经过两次池化后大小为7x7
        self.fc2 = nn.Linear(128, 10)  # MNIST有10个类别

    def forward(self, x):
        # 通过第一个卷积层和激活函数
        x = F.relu(self.conv1(x))
        # 通过第一个池化层
        x = self.pool(x)
        # 通过第二个卷积层和激活函数
        x = F.relu(self.conv2(x))
        # 通过第二个池化层
        x = self.pool(x)
        # 展平张量
        x = x.view(-1, 64 * 7 * 7)
        # 通过第一个全连接层和激活函数
        x = F.relu(self.fc1(x))
        # 通过第二个全连接层
        x = self.fc2(x)
        return x

device = torch.device("cuda")
# 实例化网络
model = SimpleCNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练网络
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs.to(device))
        # 计算损失
        loss = criterion(outputs, labels.to(device))
        # 反向传播
        loss.backward()
        # 更新权重
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # 每100个batch打印一次损失
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

# 测试网络
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

# 保存模型
torch.save(model.state_dict(), "CNN_model.pth")
