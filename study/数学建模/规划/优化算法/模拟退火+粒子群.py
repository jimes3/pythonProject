import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文
def Objective_function(X):  # 目标函数和约束条件
    X = X.flatten() #将X变为一维数组
    x1 = X[0]
    x2 = X[1]
    p1 = (max(0, 6 * x1 + 5 * x2 - 60)) ** 2
    p2 = (max(0, 10 * x1 + 20 * x2 - 150)) ** 2
    fx = 100.0 * (x2 - x1) ** 2.0 + (1 - x1) ** 2.0
    return fx + 10000 * (p1 + p2)

def main_work():
    # (变量数,粒子个数,最大迭代次数,x_min,x_max,max_vel,阈值,自身认知,群体认知,惯性因子)
    pso = PSO(2, 5, 100000, [-3,-3], [3,3], 6, 1e-4, C1=8, C2=2, W=0.1)
    fit_var_list, best_pos = pso.update_ndim()
    print("最优位置:" + str(best_pos))
    print(f"最优解为：{fit_var_list[-1]:.9f}")
    sa = SA(n=2,x_min=[-30,-30],x_max=[30,30],t1=1000,t0=1e-6,k=0.98,Markov=500,step=0.01)
    fv = sa.optimize()    #获取参数并调用函数
    #可视化
    fig, ax = plt.subplots(figsize=(15, 6))
    plt.subplot(1,2,1)
    plt.title('粒子群')
    plt.plot([i for i in range(1,len(fit_var_list)+1)],fit_var_list, color='r')
    plt.subplot(1, 2, 2)
    plt.title('模拟退火')
    plt.plot([i for i in range(1,len(fv)+1)], fv, color='r')
    plt.show()
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

class SA:
    def __init__(self, n=2,x_min=[0,0],x_max=[8,8],t1=100,t0=1e-6,k=0.98,Markov=100,step=0.5):
        # ====== 初始化随机数发生器 ======
        randseed = random.randint(1, 100)
        random.seed(randseed)  # 随机数发生器设置种子，也可以设为指定整数

        self.n = n  # 自变量数量
        self.x_min = x_min  # 给定搜索空间的下限
        self.x_max = x_max  # 给定搜索空间的上限
        self.t1 = t1  # 初始退火温度
        self.t0 = t0  # 终止退火温度，不能是0，因为只会无限趋近于0
        self.k = k  # 降温参数，T(i)=k*T(i-1)
        self.Markov = Markov  # Markov链长度，内循环运行次数
        self.step = step  # 搜索步长

    def optimize(self):
        # ====== 随机产生优化问题的初始解 ======
        xInitial = np.zeros((self.n))  # 初始化，创建数组
        for v in range(self.n):
            xInitial[v] = random.uniform(self.x_min[v], self.x_max[v])
        fxInitial = Objective_function(xInitial)

        ##################################      模拟退火算法初始化      ##############################
        xNew = np.zeros((self.n))  # 初始化，创建数组
        xNow = np.zeros((self.n))  # 初始化，创建数组
        xBest = np.zeros((self.n))  # 初始化，创建数组
        xNow[:] = xInitial[:]  # 初始化当前解，将初始解置为当前解
        xBest[:] = xInitial[:]  # 初始化最优解，将当前解置为最优解
        fxNow = fxInitial  # 将初始解的目标函数置为当前值
        fxBest = fxInitial  # 将当前解的目标函数置为最优值
        #print('初始解:{:.6f},{:.6f},\t初始值:{:.6f}'.format(xInitial[0], xInitial[1], fxInitial))

        recordIter = []  # 初始化，外循环次数
        recordFxNow = []  # 初始化，当前解的目标函数值
        recordFxBest = []  # 初始化，最佳解的目标函数值
        recordPBad = []  # 初始化，劣质解的接受概率
        n_Iter = 0  # 外循环迭代次数
        totalMar = 0  # 总计 Markov 链长度
        totalImprove = 0  # fxBest 改善次数
        nMarkov = self.Markov  # 固定长度 Markov链

        ###################################       开始模拟退火优化     ####################################
        # 外循环
        tNow = self.t1  # 当前温度
        while tNow > self.t0:  # 外循环，直到当前温度达到终止温度时结束
            kBetter = 0  # 获得优质解的次数
            kBadAccept = 0  # 接受劣质解的次数
            kBadRefuse = 0  # 拒绝劣质解的次数

            # 内循环
            for k in range(nMarkov):  # 内循环，循环次数为Markov链长度
                totalMar += 1  # 总 Markov链长度计数器

                # ---产生新解
                # 产生新解：通过在当前解附近随机扰动而产生新解，新解必须在 [min,max] 范围内
                # 方案 1：只对 n元变量中的一个进行扰动，其它 n-1个变量保持不变
                xNew[:] = xNow[:]
                v = random.randint(0, self.n - 1)  # 产生 [0,n-1]之间的随机数
                xNew[v] = xNow[v] + self.step * (self.x_max[v] - self.x_min[v]) * random.normalvariate(0, 1)
                # random.normalvariate(0, 1)：产生服从均值为0、标准差为 1 的正态分布随机实数
                xNew[v] = max(min(xNew[v], self.x_max[v]), self.x_min[v])  # 保证新解在 [min,max] 范围内

                # 计算目标函数和能量差
                fxNew = Objective_function(xNew)
                deltaE = fxNew - fxNow

                # 按 Metropolis 准则接受新解
                if fxNew < fxNow:  # 更优解：如果新解的目标函数好于当前解，则接受新解
                    accept = True
                    kBetter += 1
                else:  # 容忍解：如果新解的目标函数比当前解差，则以一定概率接受新解
                    pAccept = np.exp(-np.float64(deltaE) / np.float64(tNow))        # 计算容忍解的状态迁移概率
                    if pAccept > random.random():
                        accept = True  # 接受劣质解
                        kBadAccept += 1
                    else:
                        accept = False  # 拒绝劣质解
                        kBadRefuse += 1

                # 保存新解
                if accept == True:  # 如果接受新解，则将新解保存为当前解
                    xNow[:] = xNew[:]
                    fxNow = fxNew
                    if fxNew < fxBest:  # 如果新解的目标函数好于最优解，则将新解保存为最优解
                        fxBest = fxNew
                        xBest[:] = xNew[:]
                        totalImprove += 1
                        self.step = self.step * 0.99  # 可变搜索步长，逐步减小搜索范围，提高搜索精度

            # 完成当前温度的搜索，保存数据和输出
            pBadAccept = kBadAccept / (kBadAccept + kBadRefuse)  # 劣质解的接受概率
            recordIter.append(n_Iter)  # 当前外循环次数
            recordFxNow.append(round(fxNow, 4))  # 当前解的目标函数值
            recordFxBest.append(round(fxBest, 4))  # 最佳解的目标函数值
            recordPBad.append(round(pBadAccept, 4))  # 最佳解的目标函数值

            #if n_Iter % 10 == 0:  # 模运算，商的余数
                #print('迭代次数:{},温度:{:.3f},最优值:{:.6f}'.format(n_Iter,tNow,fxBest))

            # 缓慢降温至新的温度，降温曲线：T(k)=k*T(k-1)
            tNow = tNow * self.k
            n_Iter = n_Iter + 1
            fxBest = Objective_function(xBest)
            ###############################    结束模拟退火过程    #######################################
        print('--------------模拟退火-------------')
        print('提升次数:{:d}'.format(totalImprove))
        print("求解结果:")
        for i in range(self.n):
            print('\tx[{}] = {:.9f}'.format(i, xBest[i]))
        print('\tf(x):{:.9f}'.format(Objective_function(xBest)))
        return recordFxBest

if __name__ == '__main__':
    main_work()