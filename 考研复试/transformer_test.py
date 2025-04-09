import torch
import torch.nn as nn
import pickle
import numpy as np

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

model = Transformer(INPUT_DIM, OUTPUT_DIM, D_MODEL, NHEAD, NUM_ENCODER_LAYERS,
                    NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT)
model.load_state_dict(torch.load('transformer_model.pth'))
# 将模型设置为评估模式
model.eval()
xx = '在教育界会有一些新的教育方式'
x = []
for j in range(20):
    try:
        x.append(vocab[xx[j]])
    except KeyError:
        x.append(vocab['<UNK>'])
    except IndexError:
        x.append(vocab['<PAD>'])

# 将输入序列转换为torch.Tensor
src = torch.tensor([x], dtype=torch.long)

# 初始化解码器输入，这里可以用一个随机词开始，例如第一个词的索引
trg = torch.tensor([[x[0]]], dtype=torch.long)

# 最大预测长度
max_length = 2

# 循环预测
with torch.no_grad():
    # 编码器处理
    encoder_output = model.transformer.encoder(model.dropout(model.embedding(src) + model.positional_encoding[:src.size(1), :]).permute(1, 0, 2))
    for i in range(max_length - 1):  # 因为已经有一个初始词
        # 解码器处理
        decoder_output = model.transformer.decoder(model.dropout(model.embedding(trg) + model.positional_encoding[:trg.size(1), :]).permute(1, 0, 2), encoder_output)
        output = model.fc_out(decoder_output.permute(1, 0, 2))
        # 选择概率最大的词
        _, next_word = torch.max(output[:, -1, :], dim=1)
        next_word = next_word.unsqueeze(0)
        trg = torch.cat([trg, next_word], dim=1)

# 将预测的词索引序列转换为文本
idx2word = {idx: word for word, idx in vocab.items()}
predicted_words = []
for idx in trg.squeeze().tolist():
    predicted_words.append(idx2word[idx])

predicted_text = ''.join(predicted_words)
print("Predicted text:", predicted_text)