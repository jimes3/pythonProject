import torch
import torch.nn as nn          # 神经网络模块
import torch.nn.functional as F # 激活函数等
from torchvision import datasets, transforms # 数据集和预处理
# 设置随机种子保证可重复性
torch.manual_seed(42)

# 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),         # 将PIL图像转为Tensor (0-1范围)
    transforms.Normalize((0.1307,), (0.3081,)) # 标准化：(input - mean)/std
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

# 定义神经网络模型
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

# 加载模型参数
model = SimpleCNN()
model.load_state_dict(torch.load('CNN_model.pth'))
model.eval()  # 将模型设置为评估模式

# 推理
with torch.no_grad():  # 禁用梯度计算
    for image, label in test_dataset:
        output = model(image.unsqueeze(0))  # 前向传播, 添加批量维度
        prediction = output.argmax(dim=1).item()  # 获取预测类别
        print(f'Predicted: {prediction}, Actual: {label}')


from PIL import Image
# 加载单张图像
image_path = 'path_to_your_image.png'  # 替换为你的图像路径
image = Image.open(image_path).convert('L')  # 转换为灰度图像

# 预处理
image = transform(image).unsqueeze(0)  # 添加batch维度

# 推理
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
    print(f'Predicted: {predicted.item()}')