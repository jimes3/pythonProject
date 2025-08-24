import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import uniform
import matplotlib.pyplot as plt
n_x = 6  # 变量个数
s = np.zeros((1,n_x)).ravel()
sub = s-10 # 自变量下限
up = s+10  # 自变量上限
type = s   #-1是有理数，0是整数，1是0-1变量
# ---------------- 目标函数----------------
def objfun(X):
    f1 = np.sum((X-2)**2, axis=1)
    f2 = np.sum(X**2, axis=1)
    return np.vstack([f1, f2]).T

# ---------------- 约束函数示例 ----------------
def constraint_violation(X):
    g1 = X[:,0] + X[:,1] - 1
    g2 = X[:,0]**2 + X[:,1]**2 - 0.5
    violation = np.maximum(0, g1) + np.maximum(0, g2)
    return violation
def type_x(xx,sub,up):  #变量范围约束
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
# ---------------- 莱维飞行函数 ----------------
def levy(nPop, dim, beta=1.5):
    sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
               (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(nPop, dim) * sigma_u
    v = np.random.randn(nPop, dim)
    step = u / (np.abs(v) ** (1 / beta))
    return step

# ---------------- Deb's 可行优先法 ----------------
def dominates(x1_obj, x2_obj, x1_violation, x2_violation):
    if x1_violation == 0 and x2_violation > 0:
        return True
    elif x1_violation > 0 and x2_violation == 0:
        return False
    elif x1_violation > 0 and x2_violation > 0:
        return x1_violation < x2_violation
    else:
        return np.all(x1_obj <= x2_obj) and np.any(x1_obj < x2_obj)

# ---------------- 多目标河马算法（支持约束） ----------------
def MOHOA(SearchAgents=50, dimension=n_x, Max_iterations=50, lowerbound=sub, upperbound=up):
    X = np.random.uniform(lowerbound, upperbound, (SearchAgents, dimension))
    ObjVals = objfun(X)
    Violation = constraint_violation(X)
    # 初始Leader选择
    feasible_idx = np.where(Violation==0)[0]
    if len(feasible_idx) > 0:
        fronts = NonDominatedSorting().do(ObjVals[feasible_idx])
        Xbest = X[feasible_idx[fronts[0][0]]]
    else:
        Xbest = X[np.argmin(Violation)]
    for t in range(1, Max_iterations+1):
        # 阶段 1: 探索阶段
        for i in range(int(SearchAgents/2)):
            Dominant_hippo = Xbest
            I1 = np.random.randint(1,3)
            I2 = np.random.randint(1,3)
            Ip1 = np.random.randint(0,2,2)
            RandGroupNumber = np.random.randint(0, SearchAgents)
            RandGroup = np.random.permutation(SearchAgents)[:RandGroupNumber+1]
            MeanGroup = np.mean(X[RandGroup,:], axis=0) if len(RandGroup) > 1 else X[RandGroup[0],:]
            Alfa = {
                1: (I2 * np.random.rand(dimension) + (~Ip1[0])),
                2: 2 * np.random.rand(dimension) - 1,
                3: np.random.rand(dimension),
                4: (I1 * np.random.rand(dimension) + (~Ip1[1])),
                5: np.random.rand()
            }
            A = Alfa[np.random.randint(1,6)]
            B = Alfa[np.random.randint(1,6)]
            X_P1 = X[i,:] + np.random.rand()*(Dominant_hippo - I1*X[i,:])
            T = np.exp(-t/Max_iterations)
            if T > 0.6:
                X_P2 = X[i,:] + A*(Dominant_hippo - I2*MeanGroup)
            else:
                if np.random.rand() > 0.5:
                    X_P2 = X[i,:] + B*(MeanGroup - Dominant_hippo)
                else:
                    X_P2 = (upperbound - lowerbound) * np.random.rand(dimension) + lowerbound
            X_P1 = type_x(X_P1,sub,up)
            X_P2 = type_x(X_P2,sub,up)
            Obj_P1 = objfun(X_P1.reshape(1,-1))[0]
            Obj_P2 = objfun(X_P2.reshape(1,-1))[0]
            Violation_P1 = constraint_violation(X_P1.reshape(1,-1))[0]
            Violation_P2 = constraint_violation(X_P2.reshape(1,-1))[0]

            if dominates(Obj_P1, ObjVals[i], Violation_P1, Violation[i]):
                X[i,:] = X_P1
                ObjVals[i] = Obj_P1
                Violation[i] = Violation_P1
            if dominates(Obj_P2, ObjVals[i], Violation_P2, Violation[i]):
                X[i,:] = X_P2
                ObjVals[i] = Obj_P2
                Violation[i] = Violation_P2

        # 阶段 2: 防御阶段
        half_agents = int(SearchAgents/2)
        RL = 0.05 * levy(SearchAgents-half_agents, dimension)
        for idx, i in enumerate(range(half_agents, SearchAgents)):
            predator = lowerbound + (upperbound - lowerbound)*np.random.rand(dimension)
            distance2Leader = np.abs(predator - X[i,:])
            b = uniform.rvs(2,2)
            c = uniform.rvs(1,0.5)
            d = uniform.rvs(2,1)
            l = uniform.rvs(-2*np.pi, 4*np.pi)
            if dominates(objfun(predator.reshape(1,-1))[0], ObjVals[i],
                         constraint_violation(predator.reshape(1,-1))[0], Violation[i]):
                X_P3 = RL[idx,:]*predator + (b/(c - d*np.cos(l)))*(1/(distance2Leader+1e-9))
            else:
                X_P3 = RL[idx,:]*predator + (b/(c - d*np.cos(l)))*(1/(2*distance2Leader + np.random.rand(dimension)))
            X_P3 = type_x(X_P3,sub,up)
            Obj_P3 = objfun(X_P3.reshape(1,-1))[0]
            Violation_P3 = constraint_violation(X_P3.reshape(1,-1))[0]
            if dominates(Obj_P3, ObjVals[i], Violation_P3, Violation[i]):
                X[i,:] = X_P3
                ObjVals[i] = Obj_P3
                Violation[i] = Violation_P3

        # 阶段 3: 逃离模式
        for i in range(SearchAgents):
            LO_LOCAL = lowerbound / t
            HI_LOCAL = upperbound / t
            Alfa_local = {
                1: 2*np.random.rand(dimension)-1,
                2: np.random.rand(),
                3: np.random.randn(dimension)
            }
            D = Alfa_local[np.random.randint(1,4)]
            X_P4 = X[i,:] + np.random.rand(dimension)*(LO_LOCAL + D*(HI_LOCAL - LO_LOCAL))
            X_P4 = type_x(X_P4,sub,up)
            Obj_P4 = objfun(X_P4.reshape(1,-1))[0]
            Violation_P4 = constraint_violation(X_P4.reshape(1,-1))[0]
            if dominates(Obj_P4, ObjVals[i], Violation_P4, Violation[i]):
                X[i,:] = X_P4
                ObjVals[i] = Obj_P4
                Violation[i] = Violation_P4

        # 更新Leader
        feasible_idx = np.where(Violation==0)[0]
        if len(feasible_idx) > 0:
            fronts = NonDominatedSorting().do(ObjVals[feasible_idx])
            Xbest = X[feasible_idx[fronts[0][0]]]
        else:
            Xbest = X[np.argmin(Violation)]
        print(f"Iteration {t}: Leader Violation = {np.min(Violation)}")
    # 最终Pareto前沿（可行解）
    feasible_idx = np.where(Violation==0)[0]
    if len(feasible_idx) > 0:
        fronts = NonDominatedSorting().do(ObjVals[feasible_idx])
        return X[feasible_idx[fronts[0]]], ObjVals[feasible_idx[fronts[0]]]
    else:
        return X[np.argmin(Violation)].reshape(1,-1), ObjVals[np.argmin(Violation)].reshape(1,-1)


if __name__ == "__main__":
    pareto_solutions, pareto_objs = MOHOA(SearchAgents=200, dimension=n_x, Max_iterations=50, lowerbound=sub, upperbound=up)
    print("Pareto 前沿解：\n", pareto_solutions)
    print("对应目标函数值：\n", pareto_objs)

    # 可视化三目标 Pareto 前沿
    fig = plt.figure()
    ax = fig.add_subplot(111)#, projection='3d')
    ax.scatter(pareto_objs[:,0], pareto_objs[:,1], c='red', marker='o')
    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    #ax.set_zlabel('f3')
    ax.set_title('MOHOA Pareto Front (3 Objectives)')
    plt.show()
