import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from tqdm import tqdm

# 加载 npz 文件
npz = np.load(".\data\embedding_Tencent.npz")
#print(len(npz['embeddings']))  # 输出文件中包含的数组名称
# 打开文件并读取词汇表
with open(".\data\\vocab.pkl", "rb") as file:
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

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, dim_feedforward, dropout=0.1, embedding_matrix=None):
        super(TransformerModel, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(input_dim, model_dim)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(embedding_matrix, requires_grad=True)  # 使用预训练的词嵌入
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # 输出层
        self.fc_out = nn.Linear(model_dim, 10)  # num_classes 是标签的数量
    def forward(self, src):
        # 嵌入输入
        src_embedded = self.embedding(src) # 输出(batch_size, seq_len, model_dim)
        # 输入(seq_len, batch_size, model_dim)
        transformer_output = self.transformer_encoder(src_embedded.permute(1, 0, 2))
        # 恢复维度
        transformer_output = transformer_output.permute(1, 0, 2)
        # 取编码器的最后一个时间步或全局池化
        # 这里假设取最后一个时间步的隐藏状态
        pooled_output = transformer_output[:, -1, :]
        # 输出层
        output = self.fc_out(pooled_output)
        return output

# 参数设置
input_dim = 4762       # 词汇表大小
model_dim = 200        # 模型维度,将每个单词或标记转换为多少维的密集向量
num_heads = 4          # 编码器头数,确保 model_dim 能被 num_heads 整除
num_encoder_layers = 2 # 编码器层数,包含多头注意力层和前馈网络层
dim_feedforward = 400  # 前馈网络的维度,通常设置为 model_dim 的 2-4 倍
dropout = 0.1          # Dropout概率
num_classes = 10       # 假设有 10 个类别

device = torch.device("cuda")
# 初始化模型
model = TransformerModel(input_dim, model_dim, num_heads, num_encoder_layers, dim_feedforward,
                         dropout, embedding_matrix).to(device)
criterion = nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失,包含softmax
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        # 前向传播
        optimizer.zero_grad()
        output = model(src)  # 只传入 src
        # 计算损失
        loss = criterion(output, tgt)  # tgt 的形状是 [batch_size]
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 测试函数
def evaluate(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            # 前向传播
            output = model(src)
            _, predicted = torch.max(output.data, 1)
            total += tgt.size(0)
            correct += (predicted == tgt).sum().item()
            # 计算损失
            loss = criterion(output, tgt)
            total_loss += loss.item()
    print(f'Accuracy of the network on the 10000 test: {100 * correct / total:.2f}%')
    return total_loss / len(test_loader)
# Accuracy of the network on the 10000 test: 85.40%


# 训练和测试循环
num_epochs = 1
for epoch in tqdm(range(num_epochs)):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), "transformer_classify_model.pth")