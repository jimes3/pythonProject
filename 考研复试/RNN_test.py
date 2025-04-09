import torch
import torch.nn as nn
import pickle
import numpy as np

# 加载 npz 文件
npz = np.load("C:\\Users\jimes\PycharmProjects\课程\PyTorch框架(2022重录)\第七章：LSTM文本分类实战\\text\THUCNews\data\embedding_Tencent.npz")
#print(len(npz['embeddings']))  # 输出文件中包含的数组名称
# 打开文件并读取词汇表
with open("C:\\Users\\jimes\\PycharmProjects\\课程\\PyTorch框架(2022重录)\\第七章：LSTM文本分类实战\\text\THUCNews\data\\vocab.pkl", "rb") as file:
    vocab = pickle.load(file)

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, pad_idx, embedding_matrix=None):
        """
        初始化 RNN 网络
        vocab_size: 词汇表大小
        embedding_dim: 词嵌入维度
        hidden_dim: RNN 隐藏层维度
        num_layers:  RNN层数
        output_dim: 输出维度（类别数）
        pad_idx: <PAD> 的索引
        embedding_matrix: 预训练的词嵌入矩阵（可选）
        """
        super(SimpleRNN, self).__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(embedding_matrix, requires_grad=False)  # 使用预训练的词嵌入
        # RNN 层
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
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
        # 词嵌入
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        # RNN 处理
        output, hidden = self.rnn(embedded)  # output: [batch_size, seq_len, hidden_dim]
        # 使用 RNN 的最后一个隐藏状态
        hidden = hidden[-1]  # [batch_size, hidden_dim]
        # 全连接层
        output = self.fc(hidden)  # [batch_size, output_dim]
        # 激活函数
        return self.softmax(output)

# 超参数
vocab_size = 4762  # 词汇表大小
embedding_dim = 200  # 词嵌入维度
hidden_dim = 128  # RNN 隐藏层维度
num_layers = 2   # RNN层数
output_dim = 10  # 输出类别数
pad_idx = 4761  # <PAD> 的索引
# 词嵌入二维列表
embedding_matrix = torch.tensor(npz['embeddings'], dtype=torch.float)
# 检查词嵌入的维度
vocab_size, embedding_dim = embedding_matrix.shape
print(f"词汇表大小: {vocab_size}, 词嵌入维度: {embedding_dim}")
# 加载模型参数
model = SimpleRNN(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, pad_idx, embedding_matrix)
model.load_state_dict(torch.load('RNN_model.pth'))
# 将模型设置为评估模式
model.eval()
xx = '在教育界会有一些新的教育方式'
categories = [
    "finance",
    "realty",
    "stocks",
    "education",
    "science",
    "society",
    "politics",
    "sports",
    "game",
    "entertainment"]
x = []
for j in range(25):
    try:
        x.append(vocab[xx[j]])
    except KeyError:
        x.append(vocab['<UNK>'])
    except IndexError:
        x.append(vocab['<PAD>'])
# 预测
with torch.no_grad():
    output = model(torch.tensor(x).unsqueeze(0))
    _, predicted = torch.max(output, 1)
print(f"Predicted class: {categories[predicted.item()]}")