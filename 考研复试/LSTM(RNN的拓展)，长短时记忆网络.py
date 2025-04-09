import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from tqdm import tqdm

# 加载 npz 文件
npz = np.load("C:\\Users\jimes\PycharmProjects\课程\PyTorch框架(2022重录)\第七章：LSTM文本分类实战\\text\THUCNews\data\embedding_Tencent.npz")
#print(len(npz['embeddings']))  # 输出文件中包含的数组名称
# 打开文件并读取词汇表
with open("C:\\Users\\jimes\\PycharmProjects\\课程\\PyTorch框架(2022重录)\\第七章：LSTM文本分类实战\\text\THUCNews\data\\vocab.pkl", "rb") as file:
    vocab = pickle.load(file)

# 数据预加载
train_dataset = []
with open('./data/train.txt','r', encoding="utf-8") as file:
    while True:
        line = file.readline()  # 使用readline()逐行读取
        if not line:  # 如果读到文件末尾，line为空字符串，退出循环
            break
        columns = line.strip().split("\t")  # 使用strip()去除首尾空白字符，然后按Tab分割
        train_dataset.append(columns)
test_dataset = []
with open('./data/dev.txt','r', encoding="utf-8") as file:
    while True:
        line = file.readline()  # 使用readline()逐行读取
        if not line:  # 如果读到文件末尾，line为空字符串，退出循环
            break
        columns = line.strip().split("\t")  # 使用strip()去除首尾空白字符，然后按Tab分割
        test_dataset.append(columns)

for i in range(len(train_dataset)):
    ind1 = []
    for j in range(20):
        try:
            ind1.append(vocab[train_dataset[i][0][j]])
        except KeyError:
            ind1.append(vocab['<UNK>'])
        except IndexError:
            ind1.append(vocab['<PAD>'])
    train_dataset[i][0] = ind1

for i in range(len(test_dataset)):
    ind2 = []
    for j in range(20):
        try:
            ind2.append(vocab[test_dataset[i][0][j]])
        except KeyError:
            ind2.append(vocab['<UNK>'])
        except IndexError:
            ind2.append(vocab['<PAD>'])
    test_dataset[i][0] = ind2

class CustomDataset(Dataset):
    def __init__(self, data):
        """
        初始化数据集
        data: 原始数据集，格式为 [[features, label], ...]
        """
        self.data = data
    def __len__(self):
        #返回数据集的大小
        return len(self.data)
    def __getitem__(self, idx):
        """
        根据索引返回一个样本及其标签
        idx: 样本索引
        return: (features, label)
        """
        features, label = self.data[idx]
        # 将 features 转换为 Tensor
        features = torch.tensor(features, dtype=torch.long)
        # 将标签转换为 Tensor（假设标签是整数）
        label = torch.tensor(int(label), dtype=torch.long)
        return features, label
train_dataset = CustomDataset(train_dataset)
test_dataset = CustomDataset(test_dataset)
# 创建DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=False)
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False)

# 词嵌入二维列表
embedding_matrix = torch.tensor(npz['embeddings'], dtype=torch.float)
# 检查词嵌入的维度
vocab_size, embedding_dim = embedding_matrix.shape
print(f"词汇表大小: {vocab_size}, 词嵌入维度: {embedding_dim}")

# 定义RNN模型
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers,  pad_idx, embedding_matrix=None):
        """
        初始化 RNN 网络
        vocab_size: 词汇表大小
        embedding_dim: 词嵌入维度
        hidden_dim:  隐藏层维度
        output_dim: 输出维度（类别数）
        pad_idx: <PAD> 的索引
        embedding_matrix: 预训练的词嵌入矩阵（可选）
        """
        super(SimpleLSTM, self).__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(embedding_matrix, requires_grad=False)  # 使用预训练的词嵌入
        # LSTM 层
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)
        # 激活函数
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        """
        前向传播
        x: 输入的序列数据，形状为 [batch_size, seq_len]
        return: 输出的类别概率
        """
        # 将输入索引转换为嵌入向量
        x = self.embedding(x)  # (batch_size, sequence_length) -> (batch_size, sequence_length, embed_size)
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # 前向传播LSTM
        _, (hidden, _) = self.lstm(x, (h0, c0))  # hidden: (batch_size, sequence_length, hidden_size)
        # 使用 LSTM 的最后一个隐藏状态
        hidden = hidden[-1]  # [batch_size, hidden_dim]
        # 全连接层
        output = self.fc(hidden)  # [batch_size, output_dim]
        # 激活函数
        return self.softmax(output)

# 超参数
vocab_size = 4762  # 词汇表大小
embedding_dim = 200  # 词嵌入维度
hidden_dim = 128  #  隐藏层维度
num_layers = 2   # LSTM的层数
output_dim = 10  # 输出类别数
pad_idx = 4761  # <PAD> 的索引
#embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True) # 词嵌入不变

device = torch.device("cuda")
# 初始化模型
model = SimpleLSTM(vocab_size, embedding_dim, hidden_dim, output_dim,num_layers, pad_idx, embedding_matrix).to(device)

# 定义损失函数和优化器
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 1
for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))  # 实际对应forward函数
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

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

print(f'Accuracy of the network on the 10000 test: {100 * correct / total:.2f}%')
# Accuracy of the network on the 10000 test: 87.03%
# 保存模型
torch.save(model.state_dict(), "LSTM_model.pth")