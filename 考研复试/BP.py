import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 初始化网络参数
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return parameters

# 前向传播
def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache

# 计算损失
def compute_loss(A2, Y):
    m = Y.shape[0]
    loss = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    return loss

# 反向传播
def backward_propagation(X, Y, parameters, cache):
    m = X.shape[0]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return gradients

# 更新参数
def update_parameters(parameters, gradients, learning_rate):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return parameters

def train(X, Y, input_size, hidden_size, output_size, learning_rate, epochs):
    parameters = initialize_parameters(input_size, hidden_size, output_size)

    for i in range(epochs):
        A2, cache = forward_propagation(X, parameters)
        loss = compute_loss(A2, Y)
        gradients = backward_propagation(X, Y, parameters, cache)
        parameters = update_parameters(parameters, gradients, learning_rate)

        if i % 1000 == 0:
            print(f"Epoch {i}, Loss: {loss}")

    return parameters

def predict(X, parameters):
    A2, _ = forward_propagation(X, parameters)
    predictions = np.round(A2)
    return predictions

# 示例数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# 训练网络
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1
epochs = 10000

parameters = train(X, Y, input_size, hidden_size, output_size, learning_rate, epochs)

# 预测
predictions = predict(X, parameters)
print("Predictions:", predictions)