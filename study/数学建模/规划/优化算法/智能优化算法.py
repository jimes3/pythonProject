from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import numpy as np


class MyProblemBatch(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=3, n_constr=1,
                         xl=np.array([0,0]),
                         xu=np.array([2,2]))
    def _evaluate(self, X, out, *args, **kwargs):
        # X shape = (n_samples, n_var)
        f1 = np.sum(X**2, axis=1)
        f2 = (X[:,0]-1)**2 + X[:,1]**2
        f3 = (X[:,0]-1)**2 + X[:,1]**2
        out["F"] = np.column_stack([f1,f2,f3])

        g1 = X[:,0] + X[:,1] - 1
        out["G"] = g1.reshape(-1,1)

n_var = 2   # 决策变量数
n_obj = 3    # 目标数
problem = MyProblemBatch()

# ------------------------------生成参考方向的密度
ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=6)

# ------------------------------表示每一代有多少个候选解
algorithm_nsga3 = NSGA3(pop_size=900, ref_dirs=ref_dirs)
# 适合 >3 个目标
res_nsga3 = minimize(
    problem,
    algorithm_nsga3,
    ('n_gen', 50),#最大迭代次数
    verbose=True    # 是否显示迭代信息
)
# ------------------------------表示每一代有多少个候选解
algorithm_nsga2 = NSGA2(pop_size=900, ref_dirs=ref_dirs)
# 适合 2~3 个目标
res_nsga2 = minimize(
    problem,
    algorithm_nsga2,
    ('n_gen', 50),#最大迭代次数
    verbose=True    # 是否显示迭代信息
)


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
# 这里只画前三个目标
ax1.scatter(res_nsga3.F[:,0], res_nsga3.F[:,1], res_nsga3.F[:,2], c='r')
ax1.set_title("NSGA-III Pareto Front (3D projection)")
ax1.set_xlabel("F1")
ax1.set_ylabel("F2")
ax1.set_zlabel("F3")

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(res_nsga2.F[:,0], res_nsga2.F[:,1], c='b')
ax2.set_title("NSGA-II Pareto Front")
ax2.set_xlabel("F1")
ax2.set_ylabel("F2")

plt.tight_layout()
plt.show()
