import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# 参数
N = 3000
a, b = 0, 15
omega = (np.sqrt(5)-1)/2   # 黄金分割
K = 28.0                   # 增大 K 进入混沌


def circle_map_uniform_original(N=1000, theta0=0.1, omega=0.6180339887, K=6.0, a=0, b=10):
    theta = np.zeros(N)
    theta[0] = theta0
    for n in range(N-1):
        theta[n+1] = (theta[n] + omega - (K/(2*np.pi))*np.sin(2*np.pi*theta[n])) % 1
    # 映射到 [a, b]
    x = a + (b - a) * theta
    return x

# 改进的circle_map生成
#基于混沌反向学习和水波算法改进的白鲸优化算法---2.1混沌反向学习策略
#基于改进PSO-HO算法的无人机三维路径规划---3.3.1利用改进Circle反向学习初始化
def circle_map_uniform(N=1000, theta0=0.1, omega=(np.sqrt(5)-1)/2, K=6.0, a=0, b=10):
    theta = np.zeros(N)
    theta[0] = theta0
    #theta[0] = tent(theta0)  # 改进
    for n in range(N-1):  # 改进
        theta[n+1] = (3.5*theta[n] + omega - (K/(3.5*np.pi))*np.sin(3.5*np.pi*theta[n])) % 1
        #theta[n+1] = tent(theta[n+1])   # 改进
    # 映射到 [a, b]
    x1 = a + (b - a) * theta
    x2 = b + a - x1   # 改进
    return x1,x2

def lyapunov_exponent(N, theta0, omega, K, delta=1e-8):
    def circle_map_func(theta, omega, K):
        return (theta + omega - (K/(3.5*np.pi)) * np.sin(3.5*np.pi*theta)) % 1
    theta = theta0
    theta_perturbed = (theta0 + delta) % 1
    lyap_sum = 0

    for _ in range(N):
        theta_next = circle_map_func(theta, omega, K)
        theta_perturbed_next = circle_map_func(theta_perturbed, omega, K)

        dist = abs(theta_perturbed_next - theta_next)
        dist = dist if dist < 0.5 else 1 - dist  # 取距离最短路径

        lyap_sum += np.log(abs(dist / delta))

        # 重置扰动，使距离为delta，防止发散过大
        theta = theta_next
        theta_perturbed = (theta_next + delta * (dist/dist if dist != 0 else 1)) % 1
    return lyap_sum / N

# 迭代初试K值寻找稳定混沌的K值
K_values = np.linspace(0, 50, 50)  # 从0到10，取50个点
lyap_values = []
for K in K_values:
    lyap = lyapunov_exponent(N, 0.1, omega, K)   #最大Lyapunov指数，大于0就混沌
    lyap_values.append(lyap)

plt.figure(figsize=(10,6))
plt.plot(K_values, lyap_values, '-o', markersize=4)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Parameter K')
plt.ylabel('Maximum Lyapunov Exponent')
plt.title('Lyapunov Exponent vs K for Circle Map')
plt.grid(True)
plt.show()

######################################################################################
x_seq1,x_seq2 = circle_map_uniform(N, 0.1234, omega, K, a, b)
# 图1：时间序列散点
plt.figure(figsize=(6, 4))
plt.scatter(range(N), x_seq1, s=2, alpha=0.6)
plt.title(f"Chaotic Circle Map in [{a}, {b}]")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.show()
######################################################################################
# 以不同的初始值生成不同的随机数，范围[0,1)
x,_ = circle_map_uniform(N, 0, omega, K, a, b)
y,_ = circle_map_uniform(N, 0.5, omega, K, a, b)
z,_ = circle_map_uniform(N, 0.9, omega, K, a, b)
# 图2：随机点分布
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
# 画3D轨迹线
ax.scatter(x, y, z, lw=0.7, color='blue')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Trajectory of Circle Map with Three Variables")
plt.show()

