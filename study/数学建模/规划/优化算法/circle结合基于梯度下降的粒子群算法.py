import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文
np.set_printoptions(threshold=10) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出

def fitness(X):
    x = X.ravel() #将X变为一维数组
    return sum([100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x)-1)])

n_x = 10  # 变量个数
s = np.zeros((1,n_x)).ravel()
sub = s-100 # 自变量下限
up = s+100  # 自变量上限
type = s   #-1是有理数，0是整数，1是0-1变量
def main():
    # (变量数,粒子个数,最大迭代次数,x_min,x_max,max_vel,阈值,自身认知,群体认知,惯性因子)
    pso = PSO(n_x, 50, 1000, sub, up, 6, 1e-4, C1=2, C2=3, W=0.1)
    fit_var_list, best_pos = pso.update_ndim()
    print("最优位置:" + str(best_pos))
    print(f"最优解为：{fit_var_list[-1]:.9f}")
    #可视化
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.subplot(1,1,1)
    plt.title('粒子群')
    plt.plot([i for i in range(1,len(fit_var_list)+1)],fit_var_list, color='r')

# ===== 梯度下降 =====
def gradient_descent(f, x0, lr=1e-3, max_iter=100, tol=1e-6):
    # ===== 数值求导 =====
    def numerical_gradient_max_dir(f, x, eps=lr,mode="full"):
        x = np.array(x, dtype=float)
        grad = np.zeros_like(x)
        grads_all = np.zeros_like(x)
        for i in range(len(x)):
            x1 = x.copy()
            x2 = x.copy()
            x1[i] += eps
            x2[i] -= eps
            grads_all[i] = (f(x1) - f(x2)) / (2 * eps)
        if mode == "full":   # 多个变量进行变化，适合连续
            return grads_all
        elif mode == "max":  # 单个变量进行变化，适合离散
            idx_max = np.argmax(np.abs(grads_all))
            grad[idx_max] = grads_all[idx_max]
            return grad
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        grad = numerical_gradient_max_dir(f, x)
        #print('0',lr * grad)
        x_new = x - lr * grad
        #print(x_new)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, f(x)
def circle_map_uniform(N=30, theta0=0.1, omega=(np.sqrt(5)-1)/2, K=28.0, a=sub, b=up):
    theta = np.zeros(N)
    theta[0] = theta0  # 改进
    for n in range(N-1):  # 改进
        theta[n+1] = (3.5*theta[n] + omega - (K/(3.5*np.pi))*np.sin(3.5*np.pi*theta[n])) % 1
    # 映射到 [a, b]
    x1 = a + (b - a) * theta
    x2 = a + b - x1   # 改进
    return x1,x2
def type_x(xx):  #变量范围约束
    xx = xx.ravel()
    for v in range(len(xx)):
        if type[v] == -1:
            xx[v] = np.clip(xx[v],sub[v],up[v])
        elif type[v] == 0:
            xx[v] = np.clip(xx[v],sub[v],up[v]).round().astype(int)
        elif type[v] == 1:
            # Sigmoid 转化为概率
            xx[v] = 1 / (1 + np.exp(-xx[v]))
            # 随机取 0 或 1
            xx[v] = (np.random.rand(*xx[v].shape) < xx[v]).astype(int)
    return xx
class particle:
    # 初始化
    def __init__(self, size, max_vel, dim, ii):
        pos1,pos2 = circle_map_uniform(dim, ii/(size+1), omega=(np.sqrt(5)-1)/2, K=28.0)
        v1 = fitness(pos1)
        v2 = fitness(pos2)
        if v1 < v2:
            pos = pos1
        else:
            pos = pos2
        self.__pos = np.array(pos)  # 粒子的位置
        self.__vel = np.random.uniform(-max_vel, max_vel, (1, dim))  # 粒子的速度
        self.__bestPos = np.zeros((1, dim))  # 粒子最好的位置
        self.__fitnessValue = fitness(self.__pos)  # 适应度函数值
    #__开头的为私有属性，只在类内存在
    def set_pos(self, value):
        self.__pos = value
    def get_pos(self):
        return self.__pos
    def set_best_pos(self, value):
        self.__bestPos = value
    def get_best_pos(self):
        return self.__bestPos
    def set_vel(self, value):
        self.__vel = value
    def get_vel(self):
        return self.__vel
    def set_fitness_value(self, value):
        self.__fitnessValue = value
    def get_fitness_value(self):
        return self.__fitnessValue
class PSO:
    def __init__(self, dim, size, iter_num, x_min,x_max, max_vel, tol, best_fitness_value=float('Inf'), C1=2, C2=2, W=1):
        self.C1 = C1      #加速常数1，控制局部最优解
        self.C2 = C2      #加速常数2，控制全局最优解
        self.W = W        #惯性因子
        self.dim = dim  # 粒子的维度，变量个数
        self.size = size  # 粒子个数
        self.iter_num = iter_num  # 迭代次数
        self.x_min = x_min    #x 的下限
        self.x_max = x_max     # x 的上限
        self.max_vel = max_vel  # 粒子最大速度
        self.tol = tol  # 截止条件
        self.best_fitness_value = best_fitness_value
        self.best_position = np.zeros((1, dim))  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值
        # 对种群进行初始化
        print(particle(self.size, self.max_vel, self.dim , 3))
        self.Particle_list = [particle(self.size, self.max_vel, self.dim , i) for i in range(self.size)]
    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value
    def get_bestFitnessValue(self):
        return self.best_fitness_value
    def set_bestPosition(self, value):
        self.best_position = value
    def get_bestPosition(self):
        return self.best_position
    # 更新速度
    def update_vel(self, part):
        vel_value = self.W * part.get_vel() + self.C1 * np.random.rand() * (part.get_best_pos() - part.get_pos()) \
                    + self.C2 * np.random.rand() * (self.get_bestPosition() - part.get_pos())
        vel_value[vel_value > self.max_vel] = self.max_vel
        vel_value[vel_value < -self.max_vel] = -self.max_vel
        part.set_vel(vel_value)
    # 更新位置
    def update_pos(self, part):
        pos_value = type_x(part.get_pos() + part.get_vel())
        part.set_pos(pos_value)
        value = fitness(part.get_pos())
        if value < part.get_fitness_value():
            part.set_fitness_value(value)
            part.set_best_pos(pos_value)
        if value < self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            self.set_bestPosition(pos_value)
    #更新粒子
    def update_ndim(self):
        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
            print('第{}次最佳适应值为{}'.format(i, self.get_bestFitnessValue()))#################################################
            #if self.get_bestFitnessValue() < self.tol:
                #break
            if i % 20 == 0:
                pos_value,_ = gradient_descent(fitness, self.best_position)
                value = fitness(type_x(pos_value))
                if value < part.get_fitness_value():
                    part.set_fitness_value(value)
                    part.set_best_pos(pos_value)
                if value < self.get_bestFitnessValue():
                    self.set_bestFitnessValue(value)
                    self.set_bestPosition(pos_value)
        print('--------------粒子群--------------')
        return self.fitness_val_list, self.get_bestPosition()

if __name__ == '__main__':
    main()