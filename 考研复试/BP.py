import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 设置随机种子
torch.manual_seed(42)

# 1. 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
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

# 2. 直接使用Sequential构建网络
model = nn.Sequential(
    nn.Flatten(),                # 替代原来的x.view(-1, 784)
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练函数
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)  # 直接调用模型
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 5. 测试函数
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# 6. 执行训练测试
for epoch in range(1, 6):
    train(epoch)
    test()

# 7. 保存模型
torch.save(model.state_dict(), "mnist_model_seq.pth")