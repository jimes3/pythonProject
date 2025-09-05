import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文

##########################     导入数据      #################################
df = pd.read_csv("1.数据清洗.csv")

###########################      统计指标       #######################
start_col = 4
end_col = 1029
df['均值'] = df.iloc[:,start_col:end_col].mean(axis=1)    # 左闭右开，从0开始
df['中位数'] = df.iloc[:, start_col:end_col].median(axis=1)
df['总和'] = df.iloc[:, start_col:end_col].sum(axis=1)
df['标准差'] = df.iloc[:, start_col:end_col].std(axis=1)
df['最小值'] = df.iloc[:, start_col:end_col].min(axis=1)
df['最大值'] = df.iloc[:, start_col:end_col].max(axis=1)
df['极差'] = df['最大值'] - df['最小值']
df['变异系数'] = df['标准差'] / df['均值']
# 高级统计
df['Q1'] = df.iloc[:, start_col:end_col].quantile(0.25, axis=1)
df['Q3'] = df.iloc[:, start_col:end_col].quantile(0.75, axis=1)
df['偏度'] = df.iloc[:, start_col:end_col].skew(axis=1)
df['峰度'] = df.iloc[:, start_col:end_col].kurtosis(axis=1)
# 傅里叶变换
from scipy.fft import fft, fftfreq
# 定义一个函数：输入一行序列，返回主频
def get_main_freq(row):
    T = 1 # 采样间隔
    y = row.values
    yf = fft(y)  # 复数幅值
    xf = fftfreq(len(y), T)[:len(y)//2]  # 频率值
    amplitude = 2.0/len(y) * np.abs(yf[:len(y)//2])  # 幅值
    idx = np.argmax(amplitude[1:]) + 1
    '''plt.bar(range(20), amplitude[:20])           # 用索引画
    plt.xticks(ticks=range(0, 20, 5), labels=np.round(xf[:20:5], 3))  # 把横坐标标成频率值
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()'''
    return xf[idx]
df['主频'] = df.iloc[:, start_col:end_col].apply(get_main_freq, axis=1)

# 删除原列
df = df.drop('0（磁通密度B，T）', axis=1)  # axis=1 表示作用于每一行
df = df.drop(columns=[f'{i}' for i in range(1,1024)])

cols = df.columns.tolist()
# 假设因变量在第3列（索引2），想放到最前面
cols = [cols[2]] + cols[:2] + cols[3:]
df = df[cols]

########################      输出文件      ###########################
df = pd.DataFrame(df,columns=cols)
df.to_csv('3.时域频域特征.csv', index=False, float_format='%.6f')  # header=False


###########################      PCA主成分分析       ############################
from sklearn.decomposition import PCA  # 加载PCA算法包
#在进行主成分分析前，最好分析下自变量间的相关系数，看看有哪几大类相关性很高，
# 再来设置降维后主成分个数，后面还要对降维后的进行相关性检验，要求无相关性。
df = pd.read_csv("1.数据清洗.csv")
yy =  df['励磁波形']
y = yy.values.reshape(-1, 1)
x = df.iloc[:, 7:1031].values
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

pca = PCA(n_components=5)  # 加载PCA算法，设置降维后主成分数目看贡献到哪里比较小，再选择数目
reduced_x = pca.fit_transform(x)  # 对样本进行降维
print(reduced_x)
components = pca.components_
explained_variance_ratio = pca.explained_variance_ratio_
# 计算每个特征的综合权重
feature_weights = np.sum(np.abs(components) * explained_variance_ratio[:, np.newaxis], axis=1)
print("每个特征的权重：\n", feature_weights)

# 将降维后的数据转为 DataFrame
reduced_df = pd.DataFrame(reduced_x, columns=[f'PC{i+1}' for i in range(reduced_x.shape[1])])
final_df = pd.concat([yy.reset_index(drop=True), reduced_df], axis=1)
# 保存为新的 CSV
final_df.to_csv("3.PCA降维数据.csv", index=False)

###########################      自动编码器 (Autoencoder, AE)       ############################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出归一化到 0~1，可根据数据改
        )
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    def encode(self, x):
        return self.encoder(x)

df = pd.read_csv("1.数据清洗.csv")
yy =  df['励磁波形']
y = yy.values.reshape(-1, 1)
x = df.iloc[:, 7:1031].values
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# 将数据转换为 Tensor
X_tensor = torch.tensor(x, dtype=torch.float32)
dataset = TensorDataset(X_tensor, X_tensor)  # 输入和目标都是 X
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 潜在维度列表
k_list = [i*5 for i in range(1,11)]
errors = []
for k in k_list:
    model = Autoencoder(input_dim=X_tensor.shape[1], hidden_dim=k)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 简单训练
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, X_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    # 计算重建误差
    with torch.no_grad():
        X_hat = model(X_tensor)
        mse = ((X_tensor - X_hat)**2).mean().item()
        errors.append(mse)
# 绘制 k vs MSE 曲线
plt.figure(figsize=(6,4))
plt.plot(k_list, errors, marker='o')
plt.xlabel('潜在维度 k')
plt.ylabel('重建误差 MSE')
plt.title('潜在维度选择曲线')
plt.grid(True)
plt.show()

input_dim = x.shape[1]
hidden_dim = 25  # 压缩到几维
model = Autoencoder(input_dim, hidden_dim)
with torch.no_grad():
    X_lowdim = model.encode(X_tensor).numpy()  # (样本数, hidden_dim)
print(X_lowdim)

# 将降维后的数据转为 DataFrame
reduced_df = pd.DataFrame(X_lowdim, columns=[f'AE{i+1}' for i in range(X_lowdim.shape[1])])
final_df = pd.concat([yy.reset_index(drop=True), reduced_df], axis=1)
# 保存为新的 CSV
final_df.to_csv("3.自动编码器降维数据.csv", index=False)