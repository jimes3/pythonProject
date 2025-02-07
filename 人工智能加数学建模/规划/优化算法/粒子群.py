import numpy as np
import matplotlib.pyplot as plt


def Objective_function(X):  # 目标函数和约束条件
    X = X.flatten() #将X变为一维数组
    x1 = X[0]
    x2 = X[1]
    p1 = (max(0, 6 * x1 + 5 * x2 - 60)) ** 2
    p2 = (max(0, 10 * x1 + 20 * x2 - 150)) ** 2
    fx = 100.0 * (x2 - x1) ** 2.0 + (1 - x1) ** 2.0
    return fx + 10000 * (p1 + p2)

def main_work():
    # (变量数, 粒子个数, 最大迭代次数,x_min,x_max, max_vel, 阈值, C1=2, C2=2, W=1)
    pso = PSO(2, 5, 100000, [-30,-30], [30,30], 60, 1e-4, C1=1, C2=2, W=1)
    fit_var_list, best_pos = pso.update_ndim()
    print("最优位置:" + str(best_pos))
    print(f"最优解为：{fit_var_list[-1]:.9f}")

class particle:
    # 初始化
    def __init__(self, x_min, x_max, max_vel, dim):
        pos = np.zeros((dim))
        for i in range(0,dim):
            pos[i] = np.random.uniform(x_min[i], x_max[i])
        self.__pos = np.array(pos)  # 粒子的位置
        self.__vel = np.random.uniform(-max_vel, max_vel, (1, dim))  # 粒子的速度
        self.__bestPos = np.zeros((1, dim))  # 粒子最好的位置
        self.__fitnessValue = Objective_function(self.__pos)  # 适应度函数值
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
        self.Particle_list = [particle(self.x_min,self.x_max, self.max_vel, self.dim) for i in range(self.size)]
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
        pos_value = part.get_pos() + part.get_vel()
        part.set_pos(pos_value)
        value = Objective_function(part.get_pos())
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
            #print('第{}次最佳适应值为{}'.format(i, self.get_bestFitnessValue()))#################################################
            if self.get_bestFitnessValue() < self.tol:
                break
        print('--------------粒子群--------------')
        return self.fitness_val_list, self.get_bestPosition()
if __name__ == '__main__':
    main_work()