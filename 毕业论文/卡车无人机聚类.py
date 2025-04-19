import numpy as np
import copy

x=np.array([[33.076266, 107.060763], [32.853819, 107.11977], [32.855425, 107.02023], [32.476447, 107.24196], [32.896059, 106.851525], [32.862359, 106.920404], [33.251663, 107.247206], [33.413498, 107.192686], [33.207475, 107.171209], [33.23081, 107.169867], [33.191321, 107.379943], [33.185702, 107.522199], [33.188826, 107.616218], [33.247487, 107.715211], [33.295333, 107.597683], [33.301104, 107.643113], [33.106391, 107.707625], [33.229662, 107.5166], [33.202029, 108.01222], [33.321278, 107.509222], [32.643721, 107.428809], [33.157193, 106.698637], [33.170244, 106.753032], [33.165689, 106.890524], [33.145204, 106.846568], [33.101327, 106.451862], [33.205268, 106.681151], [33.1951, 106.956267], [33.141131, 106.6643], [33.125762, 106.743316], [33.00505, 106.697523], [33.435102, 106.56805], [33.149355, 106.624702], [32.733143, 106.235469], [32.726236, 106.431932], [33.012627, 106.178491], [32.870479, 106.4168], [32.976078, 106.472003], [33.055878, 106.301534], [33.082626, 105.970749], [33.335921, 106.155235], [33.335242, 105.966226], [33.206888, 106.42617], [33.32215, 105.822041], [33.222552, 106.083149], [33.51653, 105.911839], [33.266853, 106.254729], [33.473713, 106.043551], [32.396231, 107.752189], [32.411842, 107.830122], [32.43499, 107.553329], [32.465878, 107.483882], [32.714549, 107.885112], [32.526814, 107.570597], [32.316795, 107.820247], [32.322025, 107.996931], [32.487544, 108.089899], [32.430281, 108.009594], [32.254979, 108.116453], [32.595013, 107.717875], [32.731158, 107.991084], [32.642829, 107.753562], [32.520896, 107.907148], [32.528919, 108.19438], [32.589867, 108.039413], [32.193989, 107.960599], [32.661716, 108.204502], [32.256898, 107.899818], [33.6178, 106.92083], [33.423868, 106.993685], [33.549762, 106.985118], [33.722756, 106.955463], [33.725016, 107.05849], [33.33286, 106.9694], [33.53849, 106.91729], [33.689927, 106.855029], [33.544629, 107.82695], [33.54876, 107.993614], [33.432609, 108.086885], [33.45927, 107.97293], [33.30603, 108.04423], [33.47337, 108.11846], [33.51774, 107.98689]
            ])
needs = [8.28, 1.12, 0.76, 0.36, 2.07, 1.66, 2.89, 0.54, 1.32, 1.87, 1.3, 3.48, 1.46, 1.89, 0.7, 0.81, 1.03, 2.15, 0.62, 0.46, 0.34, 9.68, 1.21, 2.71, 1.34, 1.95, 0.36, 0.89, 3.94, 1.61, 1.01, 0.53, 1.86, 0.63, 0.36, 1.88, 1.06, 1.13, 3.1, 0.46, 6.75, 0.4, 0.72, 0.97, 0.69, 0.27, 1.01, 0.21, 0.6, 0.93, 0.97, 0.43, 0.39, 0.36, 0.49, 1.18, 1.39, 0.8, 0.67, 1.21, 0.66, 0.45, 5.09, 1.14, 1.33, 1.1, 0.83, 1.06, 1.12, 0.37, 0.35, 0.34, 0.76, 0.11, 0.25, 0.21, 0.23, 0.24, 0.11, 0.26, 0.45, 0.28, 1.09]
towns = ["七里街道",
         "法镇","小南海镇","福成镇","黄官镇","红庙镇",
         "桔园镇","双溪镇","文川镇","老庄镇",
         "马畅镇","磨子桥镇","黄安镇","槐树关镇","白石镇","长溪镇","四郎镇","戚氏镇","桑溪镇","关帝镇",
         "大河镇",
         "勉阳镇","同沟寺镇","老道寺镇","金泉镇","新铺镇","长沟河镇","褒城镇","定军山镇","温泉镇","阜川镇","张家河镇","武侯镇",
         "宁强.巴山镇","禅家岩镇","代家坝镇","铁锁关镇","胡家坝镇","大安镇","太阳岭镇",
         "略阳.城关镇","金家河镇","硖口驿镇","郭镇","白雀寺镇","西淮坝镇","接官亭镇","马蹄湾镇",
         "黎坝镇","长岭镇","简池镇","永乐镇","杨家河镇","大池镇","仁村镇","渔度镇","观音镇","小洋镇","镇巴.巴山镇","三元镇","平安镇","青水镇","泾洋镇","巴庙镇","兴隆镇","盐场镇","碾子镇","赤南镇",
         "留坝.城关镇","马道镇","武关驿镇","玉皇庙镇","江口镇","青桥驿镇","火烧店镇","留侯镇",
         "岳坝镇","长角坝镇","石墩河镇","西岔河镇","大河坝镇","陈家坝镇","袁家庄街道"]
distribution = np.array([
    [33.052466, 107.078539],
    [33.00691, 106.943519],
    [33.162452, 107.340352],
    [33.23089, 107.551097],
    [32.983209, 107.766627],
    [33.152748, 106.67331],
    [32.835302, 106.263788],
    [33.333119, 106.163146],
    [32.543287, 107.901175],
    [33.624014, 106.928095],
    [33.524357, 107.990847]
])
weights = [100, 4, 6.1, 14.5, 6, 28.6, 11,4, 16.7, 6.9, 1.7]
dis = [
    "汉中市应急管理局",
    "南郑区环境应急物资库",
    "城固县应急管理局",
    "洋县应急管理局",
    "西乡县应急管理局",
    "勉县应急管理局",
    "宁强县应急管理局",
    "略阳县应急管理局",
    "镇巴县应急管理局",
    "留坝县应急管理局",
    "佛坪县应急管理局"
]

class DBSCAN:
    def __init__(self, eps, min_samples, max_cluster_weight):
        self.eps = eps
        self.min_samples = min_samples
        self.max_cluster_weight = max_cluster_weight

    def fit(self, X, weights):
        self.labels_ = np.full(shape=X.shape[0], fill_value=-1, dtype=int)  # -1 表示未聚类
        self.cluster_id_ = 0

        for i in range(X.shape[0]):
            if self.labels_[i] != -1:
                continue
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                continue  # 不是核心点，跳过
            else:
                self._expand_cluster(X, i, neighbors, weights[i], weights)

    def _region_query(self, X, point_index):
        """找到eps邻域内的点"""
        points = X - X[point_index]
        distances = np.sqrt(np.sum(points**2, axis=1))
        neighbors = np.where(distances < self.eps)[0]
        return neighbors

    def _expand_cluster(self, X, point_index, neighbors, point_weight, weights):
        """扩展簇"""
        cluster_weight = point_weight
        self.labels_[point_index] = self.cluster_id_
        seeds = list(neighbors)

        while len(seeds) > 0:
            current_point = seeds.pop(0)
            if self.labels_[current_point] != -1:
                continue  # 已经被聚类或标记为噪声

            current_weight = weights[current_point]
            if cluster_weight + current_weight > self.max_cluster_weight:
                continue  # 超过权值和限制，不加入当前簇

            self.labels_[current_point] = self.cluster_id_
            cluster_weight += current_weight

            current_neighbors = self._region_query(X, current_point)
            if len(current_neighbors) >= self.min_samples:
                seeds.extend(current_neighbors)

        self.cluster_id_ += 1

    def plot_clusters(self, X,towns):
        for i in range(len(towns)):
            print(towns[i],self.labels_[i])
        #plt.scatter(X[:, 1], X[:, 0], c=self.labels_, cmap='Spectral', marker='o')
        #plt.title('Weighted DBSCAN Clustering')
        #plt.show()


class hs:
    def __init__(self,distribution,weights):
        self.distribution = distribution
        self.weights = weights

    def fit(self,X,Needs,Towns):
        dbscan = DBSCAN(eps=0.25, min_samples=1,max_cluster_weight=10)
        for j in range(10):
            j = 10-j
            w = 0
            res_xx = []
            res_towns0 = []
            res_needs0 = []

            x0 = copy.deepcopy(X)
            needs0 = copy.deepcopy(Needs)
            towns0 = copy.deepcopy(Towns)
            while len(x0)>0:
                a,d1,d2 = self.dist(x0,self.distribution[j],self.distribution[0])
                if w + needs0[a] > weights[j] or d2<d1:
                    x0 = self.pop(x0,a)
                    towns0.pop(a)
                    needs0.pop(a)
                    continue
                w += needs0[a]
                res_xx.append(x0[a])
                x0 = self.pop(x0,a)
                res_towns0.append(towns0[a])
                towns0.pop(a)
                res_needs0.append(needs0[a])
                needs0.pop(a)
            dbscan.fit(np.array(res_xx),res_needs0)
            dbscan.plot_clusters(np.array(res_xx),res_towns0)

            X = np.array([X[i] for i in range(len(Towns)) if Towns[i] not in res_towns0])
            Needs = [Needs[i] for i in range(len(Towns)) if Towns[i] not in res_towns0]
            Towns = [x for x in Towns if x not in res_towns0]
            print(dis[j],res_towns0)
        print(dis[0],Towns)
        dbscan.fit(X,Needs)
        dbscan.plot_clusters(X,Towns)

    def dist(self,x,distribution1,distribution0):
        d1 = np.sqrt(np.sum((x-distribution1)**2, axis=1))
        d2 = np.sqrt(np.sum((x-distribution0)**2, axis=1))
        d3 = np.sqrt(np.sum((distribution1-distribution0)**2))
        re = np.argmin(d3+d1-d2)
        d1 = d1[re]
        d2 = d2[re]
        return re,d1,d2

    def pop(self,x,index_to_pop):
        return np.concatenate((x[:index_to_pop], x[index_to_pop + 1:]))


cc = hs(distribution,weights)
cc.fit(x,needs,towns)
