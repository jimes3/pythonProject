import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import torch.optim as optim
from tqdm import tqdm

# 加载 npz 文件
npz = np.load(".\data\embedding_Tencent.npz")
# 打开文件并读取词汇表
with open(".\data\\vocab.pkl", "rb") as file:
    vocab = pickle.load(file)
categories = [
    "金融",
    "现实",
    "股票",
    "教育",
    "科学",
    "社会",
    "政治",
    "运动",
    "游戏",
    "娱乐"]
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
    train_dataset[i][1] = categories[int(train_dataset[i][1])]
    for j in range(20):
        try:
            ind1.append(vocab[train_dataset[i][0][j]])
        except KeyError:
            ind1.append(vocab['<UNK>'])
        except IndexError:
            ind1.append(vocab['<PAD>'])
    train_dataset[i][0] = ind1
    ind1 = []
    for j in range(2):
        try:
            ind1.append(vocab[train_dataset[i][1][j]])
        except KeyError:
            ind1.append(vocab['<UNK>'])
        except IndexError:
            ind1.append(vocab['<PAD>'])
    train_dataset[i][1] = ind1

for i in range(len(test_dataset)):
    ind2 = []
    test_dataset[i][1] = categories[int(test_dataset[i][1])]
    for j in range(20):
        try:
            ind2.append(vocab[test_dataset[i][0][j]])
        except KeyError:
            ind2.append(vocab['<UNK>'])
        except IndexError:
            ind2.append(vocab['<PAD>'])
    test_dataset[i][0] = ind2
    ind2 = []
    for j in range(2):
        try:
            ind2.append(vocab[test_dataset[i][1][j]])
        except KeyError:
            ind2.append(vocab['<UNK>'])
        except IndexError:
            ind2.append(vocab['<PAD>'])
    test_dataset[i][1] = ind2

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
        label = torch.tensor(label, dtype=torch.long)
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

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        '''
        INPUT_DIM,        # 源语言词汇表大小（如英语的token数量）
        OUTPUT_DIM,       # 目标语言词汇表大小（如中文的token数量）
        D_MODEL,          # 模型维度（通常512/768等，建议与词嵌入维度一致）
        NHEAD,            # 多头注意力头数（常用8/16，需能被d_model整除）
        NUM_ENCODER_LAYERS,  # 编码器层数（常用6，深层网络可能需要梯度检查点）
        NUM_DECODER_LAYERS,  # 解码器层数（通常与编码器层数相同）
        DIM_FEEDFORWARD,  # 前馈层维度（通常2048，是d_model的4倍左右）
        DROPOUT           # Dropout比率（常用0.1-0.3，防止过拟合）
        '''
        self.embedding = nn.Embedding(input_dim, d_model)
        self.positional_encoding = nn.Parameter(embedding_matrix, requires_grad=False)  # 使用预训练的词嵌入
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)  # 随机失活神经元
    def forward(self, src, trg):
        src_seq_length = src.size(1)  # 获取当前序列实际长度
        trg_seq_length = trg.size(1)
        # 输出(batch_size, seq_len, model_dim)
        src = self.dropout(self.embedding(src) + self.positional_encoding[:src_seq_length, :])
        trg = self.dropout(self.embedding(trg) + self.positional_encoding[:trg_seq_length, :])

        src = src.permute(1, 0, 2)
        trg = trg.permute(1, 0, 2)
        output = self.transformer(src, trg)# 输入(seq_len, batch_size, model_dim)

        output = self.fc_out(output.permute(1, 0, 2))

        return output

# 定义模型参数
INPUT_DIM = 4762
OUTPUT_DIM = 4762
D_MODEL = 200
NHEAD = 2
NUM_ENCODER_LAYERS = 1
NUM_DECODER_LAYERS = 1
DIM_FEEDFORWARD = 200
DROPOUT = 0.1

device = torch.device("cuda")
model = Transformer(INPUT_DIM, OUTPUT_DIM, D_MODEL, NHEAD, NUM_ENCODER_LAYERS,
                    NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT).to(device)

# 定义损失函数和优化器
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, iterator, optimizer, criterion, clip):
    model.train()  # 设置模型为训练模式
    epoch_loss = 0  # 记录每个 epoch 的总损失
    for _,(src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()  # 清空梯度
        # 前向传播
        output = model(src, trg)  # 预测输出
        # 计算损失
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)  # 展平输出
        trg = trg.view(-1)  # 展平目标（去掉第一个 token）

        loss = criterion(output, trg)  # 计算损失
        loss.backward()  # 反向传播
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # 更新参数
        epoch_loss += loss.item()  # 累加损失
    return epoch_loss / len(iterator)  # 返回平均损失

def evaluate(model, iterator, criterion):
    model.eval()  # 设置模型为评估模式
    epoch_loss = 0  # 记录每个 epoch 的总损失
    with torch.no_grad():  # 禁用梯度计算
        for _,(src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            # 前向传播
            output = model(src, trg)
            # 计算损失
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)  # 展平输出
            trg = trg.view(-1)  # 展平目标（去掉第一个 token）
            loss = criterion(output, trg)  # 计算损失
            epoch_loss += loss.item()  # 累加损失
    return epoch_loss / len(iterator)  # 返回平均损失

CLIP = 1  # 梯度裁剪的阈值
for epoch in range(1):
    # 训练
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    # 打印结果
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')

# 测试模型
test_loss = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.3f}')

# 保存模型
torch.save(model.state_dict(), "transformer_model.pth")