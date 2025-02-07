import numpy as np
from scipy.special import gammaln
from scipy.stats import uniform
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

# 调用levy函数的例子
n, m, beta = 1000, 1000, 1.5  # 例子中的参数
z = levy(n, m, beta)

def fitness(X):
    # 根据你的问题定义适应度函数
    X = X.ravel()
    a = [X[i]**2 for i in range(len(X))]
    return sum(a)

def HO(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    lowerbound = np.ones((1, dimension)) * lowerbound
    upperbound = np.ones((1, dimension)) * upperbound

    # 初始化
    X = lowerbound + (upperbound - lowerbound) * np.random.rand(SearchAgents, dimension)
    fit = np.array([fitness(L) for L in X])

    # 最优解定义
    fbest = np.min(fit)
    Xbest = X[np.argmin(fit)]

    best_so_far = np.full(Max_iterations, np.inf)
    best_so_far[0] = fbest

    # 主循环
    for t in range(1, Max_iterations):
        # 更新最优解
        f_current_best = np.min(fit)
        if f_current_best < fbest:
            fbest = f_current_best
            Xbest = X[np.argmin(fit)]

        # Phase 1: Exploration
        for i in range(int(SearchAgents / 2)):
            # Phase1: The hippopotamuses position update in the river or pond (Exploration)
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

            X_P1 = X[i, :] + np.random.rand() * (Dominant_hippopotamus - I1 * X[i, :])
            T = np.exp(-t / Max_iterations)
            if T > 0.6:
                X_P2 = X[i, :] + A * (Dominant_hippopotamus - I2 * MeanGroup)
            else:
                if np.random.rand() > 0.5:
                    X_P2 = X[i, :] + B * (MeanGroup - Dominant_hippopotamus)
                else:
                    X_P2 = (upperbound - lowerbound) * np.random.rand(dimension) + lowerbound

            X_P2 = np.clip(X_P2, lowerbound, upperbound)
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

        # Phase 2: Exploration
        # 计算中间值，避免在循环中重复计算
        half_agents = int(SearchAgents / 2)

        for i in range(half_agents + 1, SearchAgents):
            # Generate a random predator within the search space
            predator = lowerbound + (upperbound - lowerbound) * np.random.rand(dimension)

            # Calculate the fitness of the predator
            L = predator
            F_HL = fitness(L)

            # Calculate the distance from the predator to the current hippo
            distance2Leader = np.abs(predator - X[i, :])

            # Generate random numbers using uniform distribution
            b = uniform.rvs(loc=2, scale=2, size=[1, 1])[0][0]
            c = uniform.rvs(loc=1, scale=0.5, size=[1, 1])[0][0]
            d = uniform.rvs(loc=2, scale=1, size=[1, 1])[0][0]
            l = uniform.rvs(loc=-2 * np.pi, scale=4 * np.pi, size=[1, 1])[0][0]

            # Generate random numbers using levy distribution
            RL = 0.05 * levy(SearchAgents, dimension, 1.5)[i - half_agents - 1, :]

            # Update the hippo's position based on the predator
            if fit[i] > F_HL:
                X_P3 = RL * predator + (b / (c - d * np.cos(l))) * (1 / distance2Leader)
            else:
                X_P3 = RL * predator + (b / (c - d * np.cos(l))) * (1 / (2 * distance2Leader + np.random.rand(dimension)))

            # Ensure the updated position is within the bounds
            X_P3 = np.clip(X_P3, lowerbound, upperbound)

            # Calculate the fitness of the new position
            L = X_P3
            F_P3 = fitness(L)

            # Update the hippo's position if the new one is better
            if F_P3 < fit[i]:
                X[i, :] = X_P3
                fit[i] = F_P3

        # Phase 3: Exploitation
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

            # 限制X_P4在界限内
            X_P4 = np.clip(X_P4, lowerbound, upperbound)

            # 计算新位置的适应度
            L = X_P4
            F_P4 = fitness(L)

            # 如果新位置的适应度更好，则更新
            if F_P4 < fit[i]:
                X[i, :] = X_P4
                fit[i] = F_P4

        # Store the best score for the current iteration
        best_so_far[t] = fbest
        print(f'Iteration {t}: Best Cost = {best_so_far[t]}')

    return fbest, Xbest, best_so_far

SearchAgents = 50
Max_iterations = 100
lowerbound = -100
upperbound = 100
dimension = 2

Best_score, Best_pos, HO_curve = HO(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness)