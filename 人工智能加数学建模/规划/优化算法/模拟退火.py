'''参考文献：
（1）胡山鹰，陈丙珍，非线性规划问题全局优化的模拟退火法，清华大学学报，1997,37(6):5-9
（2）李歧强，具有约束指导的模拟退火算法，系统工程，2001,19(3):49-55
'''
import math
import random
import pandas as pd
import numpy as np

# 定义优化问题的目标函数，最小化问题
def Objective_function(X):  # 目标函数和约束条件
    X = X.flatten() #将X变为一维数组
    x1 = X[0]
    x2 = X[1]
    p1 = (max(0, 6 * x1 + 5 * x2 - 60)) ** 2
    p2 = (max(0, 10 * x1 + 20 * x2 - 150)) ** 2
    fx = 100.0 * (x2 - x1) ** 2.0 + (1 - x1) ** 2.0
    return fx + 10000 * (p1 + p2)
def main_work():
    sa = SA(n=2,x_min=[-30,-30],x_max=[30,30],t1=1000,t0=1e-6,k=0.98,Markov=500,step=0.01)
    sa.optimize()
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
        return n_Iter, xBest, fxBest, fxNow, recordIter, recordFxNow, recordFxBest, recordPBad



if __name__ == '__main__':
    main_work()