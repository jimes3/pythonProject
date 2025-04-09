import os
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
# 创建保存目录
save_dir = "./mnist_images"
os.makedirs(save_dir, exist_ok=True)

# 下载数据集（不进行归一化）
transform = transforms.Compose([
    transforms.ToTensor()  # 仅转换为Tensor，不归一化
])

# 下载训练集和测试集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=False,
    transform=transform
)

# 获取一个样本
image, label = train_dataset[0]  # 第0个样本
# 调整张量维度顺序：PyTorch是 (C, H, W)，Matplotlib需要 (H, W, C)
image = image.permute(1, 2, 0)
# 显示图片
plt.imshow(image)
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=False,
    transform=transform
)

# 保存训练集图片
for idx in range(len(train_dataset)):
    img, label = train_dataset[idx]

    # 转换为PIL Image
    img_pil = transforms.ToPILImage()(img)

    # 创建标签子目录
    label_dir = os.path.join(save_dir, "train", str(label))
    os.makedirs(label_dir, exist_ok=True)

    # 保存图片（格式：label_index.png）
    img_pil.save(os.path.join(label_dir, f"{label}_{idx}.png"))

# 保存测试集图片（全部）
for idx in range(len(test_dataset)):
    img, label = test_dataset[idx]

    img_pil = transforms.ToPILImage()(img)

    label_dir = os.path.join(save_dir, "test", str(label))
    os.makedirs(label_dir, exist_ok=True)

    img_pil.save(os.path.join(label_dir, f"{label}_{idx}.png"))

print(f"图片已保存至 {save_dir} 目录")