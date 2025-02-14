import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 初始化网络参数
def initialize_parameters(layer_dims):
    """
    初始化网络参数
    :param layer_dims: 列表，表示每一层的神经元数量，例如 [input_size, hidden_size1, hidden_size2, ..., output_size]
    :return: 参数字典，包含每一层的 W 和 b
    """
    np.random.seed(42)
    parameters = {}
    L = len(layer_dims)  # 网络的总层数

    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l-1], layer_dims[l]) * 0.01
        parameters[f'b{l}'] = np.zeros((1, layer_dims[l]))

    return parameters

# 前向传播
def forward_propagation(X, parameters):
    """
    前向传播
    :param X: 输入数据
    :param parameters: 参数字典
    :return: 最后一层的输出 A，以及缓存（包含每一层的 Z 和 A）
    """
    caches = []
    A = X
    L = len(parameters) // 2  # 网络的总层数（不包括输入层）

    for l in range(1, L+1):
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        Z = np.dot(A, W) + b
        A = sigmoid(Z)
        caches.append((Z, A))  # 缓存每一层的 Z 和 A

    return A, caches

# 计算损失
def compute_loss(A2, Y):
    """
    计算损失
    :param A: 最后一层的输出
    :param Y: 真实标签
    :return: 损失值
    """
    m = Y.shape[0]
    loss = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    return loss

# 反向传播
def backward_propagation(X, Y, parameters, caches):
    """
    反向传播
    :param X: 输入数据
    :param Y: 真实标签
    :param parameters: 参数字典
    :param caches: 缓存（包含每一层的 Z 和 A）
    :return: 梯度字典，包含每一层的 dW 和 db
    """
    gradients = {}
    L = len(parameters) // 2  # 网络的总层数（不包括输入层）
    m = X.shape[0]

    # 最后一层的梯度
    A = caches[-1][1]  # 最后一层的输出
    dZ = A - Y
    dW = np.dot(caches[-2][1].T, dZ) / m  # 倒数第二层的 A 是倒数第一层的输入
    db = np.sum(dZ, axis=0, keepdims=True) / m
    gradients[f'dW{L}'] = dW
    gradients[f'db{L}'] = db

    # 从倒数第二层到第一层的梯度
    for l in reversed(range(1, L)):
        dA_prev = np.dot(dZ, parameters[f'W{l+1}'].T)
        dZ = dA_prev * sigmoid_derivative(caches[l-1][1])
        dW = np.dot(caches[l-2][1].T, dZ) / m if l > 1 else np.dot(X.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        gradients[f'dW{l}'] = dW
        gradients[f'db{l}'] = db

    return gradients

# 更新参数
def update_parameters(parameters, gradients, learning_rate):
    """
    更新参数
    :param parameters: 参数字典
    :param gradients: 梯度字典
    :param learning_rate: 学习率
    :return: 更新后的参数字典
    """
    L = len(parameters) // 2

    for l in range(1, L+1):
        parameters[f'W{l}'] -= learning_rate * gradients[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * gradients[f'db{l}']

    return parameters

def train(X, Y, layer_dims, learning_rate, epochs):
    """
    训练网络
    :param X: 输入数据
    :param Y: 真实标签
    :param layer_dims: 列表，表示每一层的神经元数量
    :param learning_rate: 学习率
    :param epochs: 训练轮数
    :return: 训练后的参数字典
    """
    parameters = initialize_parameters(layer_dims)

    for i in range(epochs):
        A, caches = forward_propagation(X, parameters)
        loss = compute_loss(A, Y)
        gradients = backward_propagation(X, Y, parameters, caches)
        parameters = update_parameters(parameters, gradients, learning_rate)

        if i % 1000 == 0:
            print(f"Epoch {i}, Loss: {loss}")

    return parameters

def predict(X, parameters):
    """
    预测
    :param X: 输入数据
    :param parameters: 参数字典
    :return: 预测结果
    """
    A, _ = forward_propagation(X, parameters)
    predictions = np.round(A)
    return predictions

# 示例数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# 定义网络结构
layer_dims = [2, 4, 4, 1]  # 输入层 2 个神经元，两个隐藏层各 4 个神经元，输出层 1 个神经元

# 训练网络
learning_rate = 0.1
epochs = 10000
parameters = train(X, Y, layer_dims, learning_rate, epochs)

# 预测
predictions = predict(X, parameters)
print("Predictions:", predictions)