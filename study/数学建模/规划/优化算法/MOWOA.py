import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

#  目标函数
def objfun(x):
    f1 = x[:, 0]
    g = 1 + 9 * np.mean(x[:, 1:], axis=1)
    f2 = g * (1 - np.sqrt(f1 / g))
    f3 = np.sum(x**2, axis=1)  # axis=1 表示对每个个体的决策变量求和
    return np.vstack([f1, f2, f3]).T

#  约束函数
def constraint_violation(x):
    # x: (nPop, dim)
    #  <= 0
    g1 = x[:,0] + x[:,1] - 1
    g2 = x[:,0]**2 + x[:,1]**2 - 0.5
    violation = np.maximum(0, g1) + np.maximum(0, g2)
    return violation

#  多目标鲸鱼优化算法(MOWOA)
#                     种群大小   变量维度  迭代次数   下界   上界
def MOWOA_Constrained(nPop=50, dim=10, nGen=50, lb=0, ub=1):
    Pop = np.random.uniform(lb, ub, (nPop, dim))
    ObjVals = objfun(Pop)
    Violation = constraint_violation(Pop)
    for gen in range(1, nGen + 1):
        # 优先选择可行解
        feasible_idx = np.where(Violation == 0)[0]
        if len(feasible_idx) > 0:
            fronts = NonDominatedSorting().do(ObjVals[feasible_idx])
            Leader = Pop[feasible_idx[fronts[0]]]  # 可行解 Pareto 前沿
        else:
            Leader = Pop[np.argmin(Violation)]  # 全部不可行，选择约束最小的
        a = 2 - 2 * gen / nGen  # 收敛因子
        for i in range(nPop):
            r1, r2, r3 = np.random.rand(3)
            b = 1  # 螺旋参数
            leader = Leader[np.random.randint(len(Leader))]
            D = np.abs(leader - Pop[i])
            if np.random.rand() < 0.5:
                # 包围猎物
                Pop[i] = leader - a * D * r1
            else:
                # 螺旋更新
                A = 2 * a * r2 - a
                C = 2 * r3
                Pop[i] = leader - A * D - C * b * np.exp(b * r1) * np.cos(2 * np.pi * r1)
            # 边界处理
            Pop[i] = np.clip(Pop[i], lb, ub)
        ObjVals = objfun(Pop)
        Violation = constraint_violation(Pop)
        print(f"Iteration {gen}")
    # 最终 Pareto 前沿
    feasible_idx = np.where(Violation == 0)[0]
    if len(feasible_idx) > 0:
        fronts = NonDominatedSorting().do(ObjVals[feasible_idx])
        BestSolution = Pop[feasible_idx[fronts[0]]]
        BestObjVal = ObjVals[feasible_idx[fronts[0]]]
    else:
        idx = np.argmin(Violation)
        BestSolution = Pop[idx:idx+1]
        BestObjVal = ObjVals[idx:idx+1]
    return BestSolution, BestObjVal

if __name__ == "__main__":
    pareto_solutions, pareto_objs = MOWOA_Constrained(nPop=50, dim=10, nGen=100)
    print("Pareto 前沿解（可行解）：")
    print(pareto_solutions)
    print("对应目标函数值：")
    print(pareto_objs)

    # 可视化 Pareto 前沿
    plt.figure(figsize=(7, 5))
    plt.scatter(pareto_objs[:, 0], pareto_objs[:, 1], pareto_objs[:, 2], c='red', marker='o')
    plt.title("Pareto Front - Constrained MOWOA (ZDT1)")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(True)
    plt.show()

    # 可视化三目标 Pareto 前沿
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pareto_objs[:, 0], pareto_objs[:, 1], pareto_objs[:, 2],
               c='red', marker='o')
    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')
    ax.set_title('Pareto Front - Constrained MOWOA (3 Objectives)')
    plt.show()