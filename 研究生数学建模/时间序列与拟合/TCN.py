import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文
plt.style.use('ggplot')

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        """
        input_size: 输入特征数（多因素个数）
        output_size: 输出特征数（单因素=1，多步预测=步数）
        num_channels: 每层通道数列表，例如 [16, 32, 64]
        kernel_size: 卷积核大小
        dropout: dropout比例
        return_seq: 是否返回整个序列预测 (True=多步预测，False=单步预测)
        """
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.output_size = output_size
        self.init_weights()
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
    def forward(self, x):
        y1 = self.tcn(x)                         # (batch, channels, seq_len)
        out = self.linear(y1.transpose(1, 2))    # (batch, seq_len, output_size)
        return out[:, -self.output_size, :]                 # 取最后一步，单步预测

# 指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
output_size = 1
seq_len = 100  # 训练序列长度
# 多因素 → 单因素单步预测。通道数、输出单元数、每个卷积层的通道数、卷积核的大小
model = TCN(input_size=1, output_size=output_size, num_channels=[32, 32, 16], kernel_size=3, dropout=0).to(device)

# ---------------- 数据读取 ----------------
def series_to_XY(series, seq_len):
    series = np.array(series).astype(float)
    N = len(series)
    XX = []
    YY = []
    for i in range(0, N - seq_len):
        x = series[i:i+seq_len]
        XX.append([x.tolist()])   # [[...]] -> 1 个通道
        YY.append(series[i+seq_len])  # 下一个值作为目标，可以改成其他目标
    return XX, YY
df = pd.read_csv("时间序列预测数据集.csv")
seq = torch.from_numpy(df['Temp'].values.astype(float)).float()

scaler = StandardScaler()
series_scaled = scaler.fit_transform(seq.reshape(-1, 1))
seq = series_scaled.flatten()       # 转回 1D

train_x,train_y = series_to_XY(seq,seq_len)

# 转成 Tensor
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)

# 创建 Dataset
dataset = TensorDataset(train_x, train_y)
# 创建 DataLoader
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset, batch_size=64, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 训练
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        # x: (batch, 1, seq_len)，y: (batch,)
        optimizer.zero_grad()
        output = model(x)           # (batch, 1)
        loss = criterion(output.squeeze(), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
# 可视化
pred = model(train_x.to(device)).detach().cpu().numpy()
pred = torch.tensor(scaler.inverse_transform(pred), dtype=torch.float32)
train_y = scaler.inverse_transform(train_y.reshape(-1, 1))  # 反标准化
mse = torch.mean((pred - train_y) ** 2)
print(mse)
# 画出原始值的曲线
plt.plot(range(len(train_y)), train_y, color='k', label='y')
# 画出模型的预测线
plt.plot(range(len(pred.ravel())), pred.ravel(), 'r', label='pred')
plt.title('TCN')
plt.legend(loc='upper left')
plt.show()

# 预测
def predict(train_x,H):
    model.eval()
    preds = []
    last_seq = train_x[-1].unsqueeze(1)  # 取训练集最后一条序列作为预测起点
    print(last_seq.shape)
    # 拷贝一份最后序列
    x_input = last_seq.clone().to(device)
    for h in range(H):
        with torch.no_grad():
            y_pred = model(x_input)
            y_pred_val = y_pred.cpu().numpy().flatten()[0]
            preds.append(y_pred_val)
        # 删除最老的时间步，加上预测值
        y_pred_tensor = torch.tensor([[[y_pred_val]]], dtype=torch.float32).to(device)
        x_input = torch.cat([x_input[:, :, 1:], y_pred_tensor], dim=2)
    # 反标准化
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    print("未来预测值:", preds)
    return preds

# 稳定性分析
def wending():
    model.eval()
    with torch.no_grad():
        # 原始预测
        pred_orig = model(train_x.to(device)).detach().cpu().numpy()
        # 加噪声预测
        noise_std = 0.01
        x_noisy = train_x + torch.randn_like(train_x) * noise_std
        pred_noisy = model(x_noisy.to(device)).detach().cpu().numpy()
    # 反标准化
    pred_orig = scaler.inverse_transform(pred_orig)
    pred_noisy = scaler.inverse_transform(pred_noisy)
    # 计算 MSE / 差异
    import numpy as np
    mse_orig = np.mean((pred_orig - train_y.reshape(-1,1))**2)
    mse_noisy = np.mean((pred_noisy - train_y.reshape(-1,1))**2)
    print(f"原始 MSE: {mse_orig:.4f}")
    print(f"加噪声 MSE: {mse_noisy:.4f}")

# 不确定性分析
def buqueding(train_x):
    # 残差分析
    residuals = pred - train_y
    # 对 residuals 随机采样，否则时间太长
    sample_resid = np.random.choice(residuals.ravel(), size=3000, replace=False)
    import seaborn as sns
    sns.histplot(sample_resid, kde=True)
    plt.title("残差正态分布检验")
    plt.show()
    import statsmodels.api as sm
    sm.qqplot(sample_resid, line='45', fit=True)
    plt.title("残差QQ图")  #靠近45度线表明符合正态
    plt.show()
    from scipy.stats import shapiro,normaltest, jarque_bera
    stat, p = normaltest(residuals)
    print('D’Agostino K² test p-value:', np.round(float(p),7))
    stat, p = shapiro(residuals)
    print('Shapiro-Wilk test p-value:', np.round(float(p),7))
    stat, p = jarque_bera(residuals)
    print('Jarque-Bera test p-value:', np.round(float(p),7))
    if p > 0.05:
        print("残差近似正态")
    else:
        print("残差偏离正态")

    # bootstrap
    H = 10        # 预测步数
    B = 50       # bootstrap 次数
    sim_preds = np.zeros((B, H))
    for b in range(B):
        # 预测未来 H 步
        noise_std = 0.01
        x_noisy = train_x + torch.randn_like(train_x) * noise_std
        sim_preds[b, :] = predict(x_noisy,H)

    # 计算均值和 95% CI
    mean_pred = np.mean(sim_preds, axis=0)
    lower = np.percentile(sim_preds, 2.5, axis=0)
    upper = np.percentile(sim_preds, 97.5, axis=0)
    # 输出
    print("预测均值:", mean_pred)
    print("95% CI 下界:", lower)
    print("95% CI 上界:", upper)
    # 可视化
    plt.figure(figsize=(8,5))
    plt.plot(range(H), mean_pred, color='blue', label='预测均值')
    plt.fill_between(range(H), lower, upper, color='blue', alpha=0.2, label='95% CI')
    plt.xlabel("步数")
    plt.ylabel("预测值")
    plt.title("Bootstrap 预测均值与 95% CI")
    plt.legend()
    plt.show()

#predict(train_x,10)
#wending()
buqueding(train_x)