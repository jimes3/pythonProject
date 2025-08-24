import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import matplotlib.pyplot as plt
n_x = 6  # 变量个数
s = np.zeros((1,n_x)).ravel()
sub = s-10 # 自变量下限
up = s+10  # 自变量上限
type = s   #-1是有理数，0是整数，1是0-1变量

# ---------------- 目标函数 ----------------
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

# ---------------- 多目标鲸鱼算法（MOWOA） ----------------
def MOWOA(SearchAgents=50, dimension=n_x, Max_iterations=50, lb=sub, ub=up):
    X = np.random.uniform(lb, ub, (SearchAgents, dimension))
    ObjVals = objfun(X)
    Violation = constraint_violation(X)

    # 初始Leader选择（可行解前沿，若无可行解，取最小违反度）
    feasible_idx = np.where(Violation==0)[0]
    if len(feasible_idx) > 0:
        fronts = NonDominatedSorting().do(ObjVals[feasible_idx])
        Leader = X[feasible_idx[fronts[0][0]]]  # Leader 选第一个前沿
    else:
        Leader = X[np.argmin(Violation)]

    for t in range(1, Max_iterations+1):
        a = 2 - 2 * t / Max_iterations  # 收敛因子
        for i in range(SearchAgents):
            r1, r2 = np.random.rand(), np.random.rand()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = np.random.rand()
            l = np.random.uniform(-1,1)
            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C * Leader - X[i,:])
                    X_new = Leader - A * D
                else:
                    rand_idx = np.random.randint(0, SearchAgents)
                    X_rand = X[rand_idx,:]
                    D = abs(C * X_rand - X[i,:])
                    X_new = X_rand - A * D
            else:
                # 螺旋更新
                b = 1
                D = abs(Leader - X[i,:])
                X_new = D * np.exp(b*l) * np.cos(2*np.pi*l) + Leader
            X_new = type_x(X_new,sub,up)
            Obj_new = objfun(X_new.reshape(1,-1))[0]
            Violation_new = constraint_violation(X_new.reshape(1,-1))[0]
            # 更新个体
            if dominates(Obj_new, ObjVals[i], Violation_new, Violation[i]):
                X[i,:] = X_new
                ObjVals[i] = Obj_new
                Violation[i] = Violation_new
        # 更新Leader
        feasible_idx = np.where(Violation==0)[0]
        if len(feasible_idx) > 0:
            fronts = NonDominatedSorting().do(ObjVals[feasible_idx])
            # 选拥挤距离最大的前沿个体作为Leader更均匀，可选此处简单选第一个
            Leader = X[feasible_idx[fronts[0][0]]]
        else:
            Leader = X[np.argmin(Violation)]
        print(f"Iteration {t}: Leader Violation = {np.min(Violation)}")
    # 返回Pareto前沿（所有可行解）
    feasible_idx = np.where(Violation==0)[0]
    if len(feasible_idx) > 0:
        fronts = NonDominatedSorting().do(ObjVals[feasible_idx])
        return X[feasible_idx[fronts[0]]], ObjVals[feasible_idx[fronts[0]]]
    else:
        # 全部不可行，返回违反度最小的解
        best_idx = np.argmin(Violation)
        return X[best_idx].reshape(1,-1), ObjVals[best_idx].reshape(1,-1)

if __name__ == "__main__":
    pareto_solutions, pareto_objs = MOWOA(SearchAgents=500, dimension=n_x, Max_iterations=50, lb=sub, ub=up)
    print("Pareto 前沿解：\n", pareto_solutions)
    print("对应目标函数值：\n", pareto_objs)

    # 可视化三目标 Pareto 前沿
    fig = plt.figure()
    ax = fig.add_subplot(111)#, projection='3d')
    ax.scatter(pareto_objs[:,0], pareto_objs[:,1], c='blue', marker='o')
    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    #ax.set_zlabel('f3')
    ax.set_title('MOWOA Pareto Front (3 Objectives)')
    plt.show()
