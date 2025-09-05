import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文

x = np.arange(5)
y = (25, 32, 34, 20, 25)
y_offset = (3, 5, 2, 3, 3)
plt.errorbar(x, y, yerr=y_offset, capsize=3, capthick=2,ecolor='k',elinewidth=1,
             mec='k',mew=1,ms=10,alpha=1,label="Observation")
plt.show()

# 1. 雷达图
labels = np.array(["指标A", "指标B", "指标C", "指标D", "指标E"])
stats = [0.8, 0.6, 0.7, 0.9, 0.65]

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
stats = np.concatenate((stats,[stats[0]]))
angles = np.concatenate((angles,[angles[0]]))

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Radar chart
ax_radar = plt.subplot(221, polar=True)
ax_radar.plot(angles, stats, 'o-', linewidth=2)
ax_radar.fill(angles, stats, alpha=0.25)
ax_radar.set_thetagrids(angles[:-1] * 180/np.pi, labels)
ax_radar.set_title("雷达图", fontsize=14)

# 2. 热力图
data = np.random.rand(6,6)
sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu", ax=axs[0,1])
axs[0,1].set_title("热力图", fontsize=14)

# 3. 敏感性 Tornado 图
factors = ["参数A","参数B","参数C","参数D","参数E"]
impacts = [0.3, 0.6, 0.2, 0.8, 0.5]
sorted_idx = np.argsort(impacts)
axs[1,0].barh(np.array(factors)[sorted_idx], np.array(impacts)[sorted_idx], color="skyblue")
axs[1,0].set_title("敏感性 Tornado 图", fontsize=14)
axs[1,0].set_xlabel("影响程度")

# 4. Pareto 前沿图
points = np.random.rand(30, 2)
pareto = []
for i, p in enumerate(points):
    if not np.any((points[:,0] <= p[0]) & (points[:,1] <= p[1]) & ((points[:,0] < p[0]) | (points[:,1] < p[1]))):
        pareto.append(p)
pareto = np.array(pareto)

axs[1,1].scatter(points[:,0], points[:,1], c="grey", label="解空间")
axs[1,1].scatter(pareto[:,0], pareto[:,1], c="red", label="Pareto前沿")
axs[1,1].set_title("Pareto前沿图", fontsize=14)
axs[1,1].set_xlabel("目标1")
axs[1,1].set_ylabel("目标2")
axs[1,1].legend()

plt.tight_layout()
plt.show()
