import torch
import torch.nn as nn
import pickle
import numpy as np

# 加载 npz 文件
npz = np.load(".\data\embedding_Tencent.npz")
# 打开文件并读取词汇表
with open(".\data\\vocab.pkl", "rb") as file:
    vocab = pickle.load(file)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, dim_feedforward, dropout=0.1, embedding_matrix=None):
        super(TransformerModel, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(input_dim, model_dim)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(embedding_matrix, requires_grad=False)  # 使用预训练的词嵌入
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

# 词嵌入二维列表
embedding_matrix = torch.tensor(npz['embeddings'], dtype=torch.float)
# 检查词嵌入的维度
vocab_size, embedding_dim = embedding_matrix.shape
print(f"词汇表大小: {vocab_size}, 词嵌入维度: {embedding_dim}")

# 初始化模型
model = TransformerModel(input_dim, model_dim, num_heads, num_encoder_layers, dim_feedforward, dropout, embedding_matrix)
model.load_state_dict(torch.load('transformer_classify_model.pth'))
# 将模型设置为评估模式
model.eval()
xx = '中国和美国关系不是很好'
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
for j in range(20):
    try:
        x.append(vocab[xx[j]])
    except KeyError:
        x.append(vocab['<UNK>'])
    except IndexError:
        x.append(vocab['<PAD>'])

# 预测
with torch.no_grad():
    output = model(torch.tensor(x).unsqueeze(0))
    print(output)
    _, predicted = torch.max(output, 1)
print(f"Predicted class: {categories[predicted.item()]}")