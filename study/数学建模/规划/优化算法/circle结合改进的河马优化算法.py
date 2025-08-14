import numpy as np
from scipy.special import gammaln
from scipy.stats import uniform
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文

def fitness(X):
    x = X.flatten() #将X变为一维数组
    return sum([100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x)-1)])

n_x = 10  # 变量个数
s = np.zeros((1,n_x)).ravel()
sub = s-10 # 自变量下限
up = s+10  # 自变量上限
type = s   #-1是有理数，0是整数，1是0-1变量
def main():
    Best_score, Best_pos, HO_curve = HO(30, 100, sub, up, n_x, fitness)
    plt.title('河马算法')
    plt.plot(range(1,len(HO_curve)+1),HO_curve, color='r')
    plt.show()


def circle_map_uniform(N=30, theta0=0.1, omega=(np.sqrt(5)-1)/2, K=40.0, ss = s, a=sub, b=up):
    def tent(x):   # 改进，基于混沌反向学习和水波算法改进的白鲸优化算法，2.1
        if x<0.499:
            x = x/0.499
        else:
            x = (1-x)/(1-0.499)
        return x
    theta = np.zeros(N)
    theta[0] = tent(theta0)
    for n in range(N-1):
        theta[n+1] = (3.5*theta[n] + omega - (K/(3.5*np.pi))*np.sin(3.5*np.pi*theta[n])) % 1
        theta[n+1] = tent(theta[n+1])
    # 映射到 [a, b]
    x = a + (b - a) * theta
    x = a + b - x   # 改进，基于混沌反向学习和水波算法改进的白鲸优化算法，2.1
    return x
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
def HO(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    def levy(n, m, beta):
        # 计算gamma函数的对数，因为gamma函数的值可能非常大
        num = np.exp(gammaln(1 + beta) - np.pi * beta / 2.0)
        den = np.exp(gammaln((1 + beta) / 2) + (beta - 1) / 2.0 * np.log(2))
        sigma_u = num / den  # 计算标准差
        # 生成正态分布的随机数
        u = np.random.normal(0, sigma_u, (n, m))
        v = np.random.normal(0, 1, (n, m))
        # 计算levy飞行分布
        z = u / np.abs(v) ** (1.0 / beta)
        return z
    # 初始化
    X = np.zeros([SearchAgents, dimension])
    for s in range(SearchAgents):
        X[s, :] = circle_map_uniform(dimension, s/(SearchAgents+1), omega=(np.sqrt(5)-1)/2, K=40.0)
        X[s, :] = type_x(X[s, :])
    fit = np.array([fitness(L) for L in X])
    # 最优解定义
    fbest = np.min(fit)
    Xbest = X[np.argmin(fit)]

    best_so_far = np.full(Max_iterations+1, np.inf)
    best_so_far[0] = fbest
    # 主循环
    for t in range(1, Max_iterations+1):
        # Phase 1: 探索阶段
        for i in range(int(SearchAgents / 2)):
            Dominant_hippopotamus = Xbest
            I1 = np.random.randint(1, 3)
            I2 = np.random.randint(1, 3)
            Ip1 = np.random.randint(0, 2, 2)
            RandGroupNumber = np.random.randint(0, SearchAgents)
            RandGroup = np.random.permutation(SearchAgents)[:RandGroupNumber+1]  # 需要+1因为randperm是左闭右开区间
            MeanGroup = np.mean(X[RandGroup, :], axis=0) * (len(RandGroup) != 1) + X[RandGroup[0], :] * (len(RandGroup) == 1)
            Alfa = {
                1: (I2 * np.random.rand(dimension) + (~Ip1[0])),
                2: 2 * np.random.rand(dimension) - 1,
                3: np.random.rand(dimension),
                4: (I1 * np.random.rand(dimension) + (~Ip1[1])),
                5: np.random.rand()
            }
            A = Alfa[np.random.randint(1, 6)]
            B = Alfa[np.random.randint(1, 6)]

            # 改进点：自适应权重策略w    #####################################
            w = np.exp(-(t/Max_iterations)**0.5)
            X_P1 = w * X[i, :] + np.random.rand() * (Dominant_hippopotamus - I1 * X[i, :])
            T = np.exp(-t / Max_iterations)
            if T > 0.6:
                X_P2 = X[i, :] + A * (Dominant_hippopotamus - I2 * MeanGroup)
            else:
                if np.random.rand() > 0.5:
                    X_P2 = X[i, :] + B * (MeanGroup - Dominant_hippopotamus)
                else:
                    X_P2 = (upperbound - lowerbound) * np.random.rand(dimension) + lowerbound
            X_P2 = type_x(X_P2)
            X_P1 = type_x(X_P1)
            L = X_P1
            F_P1 = fitness(L)
            if F_P1 < fit[i]:
                X[i, :] = X_P1
                fit[i] = F_P1
            L2 = X_P2
            F_P2 = fitness(L2)
            if F_P2 < fit[i]:
                X[i, :] = X_P2
                fit[i] = F_P2
        # Phase 2: 防御阶段
        # 计算中间值，避免在循环中重复计算
        half_agents = int(SearchAgents / 2)
        for i in range(half_agents + 1, SearchAgents):
            # 在搜索空间内生成一个随机捕食者
            predator = lowerbound + (upperbound - lowerbound) * np.random.rand(dimension)
            # 计算捕食者的适应度
            L = predator
            F_HL = fitness(L)
            # 计算掠食者到当前河马的距离
            distance2Leader = np.abs(predator - X[i, :])
            # 使用均匀分布生成随机数
            b = uniform.rvs(loc=2, scale=2, size=[1, 1])[0][0]
            c = uniform.rvs(loc=1, scale=0.5, size=[1, 1])[0][0]
            d = uniform.rvs(loc=2, scale=1, size=[1, 1])[0][0]
            l = uniform.rvs(loc=-2 * np.pi, scale=4 * np.pi, size=[1, 1])[0][0]
            # 使用莱维分布生成随机数
            RL = 0.05 * levy(SearchAgents, dimension, 1.5)[i - half_agents - 1, :]
            # 根据捕食者更新河马的位置
            if fit[i] > F_HL:
                X_P3 = RL * predator + (b / (c - d * np.cos(l))) * (1 / distance2Leader)
            else:
                X_P3 = RL * predator + (b / (c - d * np.cos(l))) * (1 / (2 * distance2Leader + np.random.rand(dimension)))
            # 确保更新后的位置在边界内
            X_P3 = type_x(X_P3)
            # 计算新位置的适应度
            L = X_P3
            F_P3 = fitness(L)
            # 如果新的位置更好，更新河马的位置
            if F_P3 < fit[i]:
                X[i, :] = X_P3
                fit[i] = F_P3
        # Phase 3: 逃离模式
        for i in range(SearchAgents):
            # 计算局部搜索范围
            LO_LOCAL = lowerbound / t
            HI_LOCAL = upperbound / t
            # 随机生成Alfa值
            Alfa = {
                1: 2 * np.random.rand(dimension) - 1,  # 均匀分布[-1, 1)
                2: np.random.rand(),                   # 均匀分布[0, 1)
                3: np.random.randn(dimension)           # 标准正态分布
            }
            # 随机选择一个Alfa
            D = Alfa[np.random.randint(1, 4)]
            # 更新X_P4位置
            X_P4 = X[i, :] + np.random.rand() * (LO_LOCAL + D * (HI_LOCAL - LO_LOCAL))
            # 改进点  ###############################################
            k = t/Max_iterations
            X_P4 = (sub+up)/2*(1+k) - X_P4/k
            # 限制X_P4在界限内
            X_P4 = type_x(X_P4)
            # 计算新位置的适应度
            L = X_P4
            F_P4 = fitness(L)
            # 如果新位置的适应度更好，则更新
            if F_P4 < fit[i]:
                X[i, :] = X_P4
                fit[i] = F_P4
        # 存储最佳适应值
        best_so_far[t] = fbest
        print(f'Iteration {t}: Best Cost = {best_so_far[t]}')
        # 更新最优解
        f_current_best = np.min(fit)
        if f_current_best < fbest:
            fbest = f_current_best
            Xbest = X[np.argmin(fit)]
    print("最优变量为：",Xbest)
    return fbest, Xbest, best_so_far

if __name__=='__main__':
    main()