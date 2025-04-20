import numpy as np
from numpy import random
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import groupby
import warnings
#np.random.seed(36)
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文

xxx = {
    "七里街道": [33.076266, 107.060763],
    "法镇": [32.853819, 107.11977],
    "小南海镇": [32.855425, 107.02023],
    "福成镇": [32.476447, 107.24196],
    "黄官镇": [32.896059, 106.851525],
    "红庙镇": [32.862359, 106.920404],
    "桔园镇": [33.251663, 107.247206],
    "双溪镇": [33.413498, 107.192686],
    "文川镇": [33.207475, 107.171209],
    "老庄镇": [33.23081, 107.169867],
    "马畅镇": [33.191321, 107.379943],
    "磨子桥镇": [33.185702, 107.522199],
    "黄安镇": [33.188826, 107.616218],
    "槐树关镇": [33.247487, 107.715211],
    "白石镇": [33.295333, 107.597683],
    "长溪镇": [33.301104, 107.643113],
    "四郎镇": [33.106391, 107.707625],
    "戚氏镇": [33.229662, 107.5166],
    "桑溪镇": [33.202029, 108.01222],
    "关帝镇": [33.321278, 107.509222],
    "大河镇": [32.643721, 107.428809],
    "勉阳镇": [33.157193, 106.698637],
    "同沟寺镇": [33.170244, 106.753032],
    "老道寺镇": [33.165689, 106.890524],
    "金泉镇": [33.145204, 106.846568],
    "新铺镇": [33.101327, 106.451862],
    "长沟河镇": [33.205268, 106.681151],
    "褒城镇": [33.1951, 106.956267],
    "定军山镇": [33.141131, 106.6643],
    "温泉镇": [33.125762, 106.743316],
    "阜川镇": [33.00505, 106.697523],
    "张家河镇": [33.435102, 106.56805],
    "武侯镇": [33.149355, 106.624702],
    "宁强.巴山镇": [32.733143, 106.235469],
    "禅家岩镇": [32.726236, 106.431932],
    "代家坝镇": [33.012627, 106.178491],
    "铁锁关镇": [32.870479, 106.4168],
    "胡家坝镇": [32.976078, 106.472003],
    "大安镇": [33.055878, 106.301534],
    "太阳岭镇": [33.082626, 105.970749],
    "略阳.城关镇": [33.335921, 106.155235],
    "金家河镇": [33.335242, 105.966226],
    "硖口驿镇": [33.206888, 106.42617],
    "郭镇": [33.32215, 105.822041],
    "白雀寺镇": [33.222552, 106.083149],
    "西淮坝镇": [33.51653, 105.911839],
    "接官亭镇": [33.266853, 106.254729],
    "马蹄湾镇": [33.473713, 106.043551],
    "黎坝镇": [32.396231, 107.752189],
    "长岭镇": [32.411842, 107.830122],
    "简池镇": [32.43499, 107.553329],
    "永乐镇": [32.465878, 107.483882],
    "杨家河镇": [32.714549, 107.885112],
    "大池镇": [32.526814, 107.570597],
    "仁村镇": [32.316795, 107.820247],
    "渔度镇": [32.322025, 107.996931],
    "观音镇": [32.487544, 108.089899],
    "小洋镇": [32.430281, 108.009594],
    "镇巴.巴山镇": [32.254979, 108.116453],
    "三元镇": [32.595013, 107.717875],
    "平安镇": [32.731158, 107.991084],
    "青水镇": [32.642829, 107.753562],
    "泾洋镇": [32.520896, 107.907148],
    "巴庙镇": [32.528919, 108.19438],
    "兴隆镇": [32.589867, 108.039413],
    "盐场镇": [32.193989, 107.960599],
    "碾子镇": [32.661716, 108.204502],
    "赤南镇": [32.256898, 107.899818],
    "留坝.城关镇": [33.6178, 106.92083],
    "马道镇": [33.423868, 106.993685],
    "武关驿镇": [33.549762, 106.985118],
    "玉皇庙镇": [33.722756, 106.955463],
    "江口镇": [33.725016, 107.05849],
    "青桥驿镇": [33.33286, 106.9694],
    "火烧店镇": [33.53849, 106.91729],
    "留侯镇": [33.689927, 106.855029],
    "岳坝镇": [33.544629, 107.82695],
    "长角坝镇": [33.54876, 107.993614],
    "石墩河镇": [33.432609, 108.086885],
    "西岔河镇": [33.45927, 107.97293],
    "大河坝镇": [33.30603, 108.04423],
    "陈家坝镇": [33.47337, 108.11846],
    "袁家庄街道": [33.51774, 107.98689],
    "配送点":[]
}
needs = [8.28, 1.12, 0.76, 0.36, 2.07, 1.66, 2.89, 0.54, 1.32, 1.87, 1.3, 3.48, 1.46, 1.89, 0.7, 0.81, 1.03, 2.15, 0.62, 0.46,
         0.34, 9.68, 1.21, 2.71, 1.34, 1.95, 0.36, 0.89, 3.94, 1.61, 1.01, 0.53, 1.86, 0.63, 0.36, 1.88, 1.06, 1.13, 3.1, 0.46,
         6.75, 0.4, 0.72, 0.97, 0.69, 0.27, 1.01, 0.21, 0.6, 0.93, 0.97, 0.43, 0.39, 0.36, 0.49, 1.18, 1.39, 0.8, 0.67, 1.21,
         0.66, 0.45, 5.09, 1.14, 1.33, 1.1, 0.83, 1.06, 1.12, 0.37, 0.35, 0.34, 0.76, 0.11, 0.25, 0.21, 0.23, 0.24, 0.11, 0.26,
         0.45, 0.28, 1.09]
def haversine_distance(coord1, coord2):
    # 将经纬度从度数转换为弧度
    lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
    lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])
    # Haversine 公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    # 地球半径（单位：公里）
    radius = 6371.0
    # 计算距离
    distance = radius * c
    return distance

def find_consecutive_ones(nums):
    groups = []
    for k, g in groupby(enumerate(nums), key=lambda x: x[1]):
        if k == 1:
            group = list(g)
            start = group[0][0]
            end = group[-1][0]
            groups.append((start, end))

    max_length = max((end - start + 1) for start, end in groups) if groups else 0
    return groups, max_length

def calculate_weighted_delivery_time(truck_route, drone_missions,  speed_truck, speed_drone, service_time):
    # 初始化时间记录
    node_service_times = {node: 0 for node in truck_route + [m[2] for m in drone_missions]}
    weighted_time = 0
    node_weights = {i+1:values[i+1] for i in range(len(values)-1)}
    # 1. 计算卡车路径上各节点的服务时间
    current_time = 0
    for i in range(1, len(truck_route)):
        prev_node = truck_route[i-1]
        curr_node = truck_route[i]

        # 卡车行驶时间 (简化距离为1)
        travel_time = dist_matrix[prev_node,curr_node] / speed_truck
        current_time += travel_time

        # 记录节点服务完成时间 (考虑权重)
        service_completion = current_time + service_time
        node_service_times[curr_node] = service_completion
        weighted_time += service_completion * node_weights.get(curr_node, 1)

        current_time = service_completion

    # 2. 计算无人机服务的客户点时间
    for mission in drone_missions:
        launch_node, return_node, customer = mission

        # 无人机只能在卡车到达发射节点后才能发射
        launch_time = node_service_times[launch_node]

        # 无人机飞行时间 (简化为发射点到客户和客户到返回点各距离1)
        flight_time = (1 + 1) / speed_drone
        customer_service_time = launch_time + (1 / speed_drone)  # 到达客户时间
        return_time = launch_time + flight_time

        # 无人机必须在卡车到达返回节点后才能回收
        if return_time > node_service_times[return_node]:
            # 需要等待卡车到达返回节点
            delay = return_time - node_service_times[return_node]
            weighted_time += delay * node_weights.get(return_node, 1)
            return_time = node_service_times[return_node]  # 实际回收时间

        # 记录客户点服务完成时间 (考虑权重)
        node_service_times[customer] = customer_service_time
        weighted_time += customer_service_time * node_weights.get(customer, 1)

    return weighted_time, node_service_times

def find_index(lst, element):
    """
    返回元素在列表中第一次出现的索引
    如果元素不在列表中，返回 -1
    """
    try:
        return lst.index(element)
    except ValueError:
        return -1

towns_str = "白雀寺镇,太阳岭镇,代家坝镇"
towns_list = towns_str.split(',')
towns_loc = []
for i in range(len(towns_list)):
    towns_loc.append(xxx[towns_list[i]])
distribution = [
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
]
#towns_loc1 = [distribution[0]]+towns_loc   # 汉中市
#towns_loc1 = [distribution[1]]+towns_loc   # 南郑区
#towns_loc1 = [distribution[2]]+towns_loc   # 城固县
#towns_loc1 = [distribution[3]]+towns_loc   # 洋县
#towns_loc1 = [distribution[4]]+towns_loc   # 西乡县
#towns_loc1 = [distribution[5]]+towns_loc   # 勉县
#towns_loc1 = [distribution[6]]+towns_loc   # 宁强县
towns_loc1 = [distribution[7]]+towns_loc   # 略阳县
#towns_loc1 = [distribution[8]]+towns_loc   # 镇巴县
#towns_loc1 = [distribution[9]]+towns_loc   # 留坝县
#towns_loc1 = [distribution[10]]+towns_loc   # 佛坪县

coordinates = np.array(towns_loc1)
#print(towns_loc1)
values = [0]
for i in range(len(towns_list)):
    index = list(xxx.keys()).index(towns_list[i])
    values.append(needs[index])
print('权重',values)
# 初始化距离矩阵
num_points = len(coordinates)
dist_matrix = np.zeros((num_points, num_points))
dist_matrix[:,:]=np.inf
# 计算距离矩阵
for i in range(num_points):
    for j in range(num_points):
        dist_matrix[i, j] = haversine_distance(coordinates[i], coordinates[j])
#print(dist_matrix)

def fun(X,h=0):
    x = X.flatten() #将X变为一维数组
    l = int(len(x)/4)
    point=[i+1 for i in range(l)]
    def ddd(i):
        a = point[int(x[i])]
        del point[int(x[i])]
        return a
    trace = [ddd(i) for i in range(l)] #无人机卡车综合路径
    truck_route = []
    for i in range(l):#获得车辆路径
        if x[i+l]==0: # 卡车
            truck_route.append(trace[i])
    drone_missions = [(int(x[2*i]),int(x[2*i+1]),int(trace[i-l])) for i in range(l,2*l) if x[i]==1]  # 无人机任务
    truck_route = [0] + truck_route + [0]
    total_time, service_times = calculate_weighted_delivery_time(truck_route, drone_missions, 60, 150, 0)
    if h == 1:
        print('卡车',truck_route)
        print('无人机',drone_missions)
        for node, time in service_times.items():
            if node==0:continue
            node = list(xxx.keys()).index(towns_list[node-1])
            print(f"节点所需时间 {node+1}: {time:.3f}")
        for i in range(1,len(truck_route)-1):
            index = list(xxx.keys()).index(towns_list[int(truck_route[i])-1])
            print(index+1,end="-")
        towns_list.append("配送点") #防止起飞到达为配送点显示失误
        for i in range(len(drone_missions)):
            index1 = list(xxx.keys()).index(towns_list[int(drone_missions[i][0])-1])
            index2 = list(xxx.keys()).index(towns_list[int(drone_missions[i][1])-1])
            index3 = list(xxx.keys()).index(towns_list[int(drone_missions[i][2])-1])
            if index1==83:index1=-1
            if index2==83:index2=-1
            if index3==83:index3=-1
            drone = (index1+1,index2+1,index3+1)
            print(drone,end="-")
    # 约束无人机的顺序
    dr = []
    for i in range(len(drone_missions)):
        try:
            dr.append(truck_route.index(drone_missions[i][0]))
            dr.append(truck_route.index(drone_missions[i][1]))
        except ValueError:
            total_time += 10000
        if values[drone_missions[i][2]]>1: # 约束无人机载重
            total_time += 10000
        if drone_missions[i][0]==drone_missions[i][1]:
            total_time += 10000
        if drone_missions[i][0]==drone_missions[i][2]:
            total_time += 10000
        if drone_missions[i][1]==drone_missions[i][2]:
            total_time += 10000
        if dist_matrix[drone_missions[i][0],drone_missions[i][1]]+dist_matrix[drone_missions[i][1],drone_missions[i][2]]>1000:
            total_time += 10000
    for i in range(len(dr)):
        if dr[-1]==0:
            del dr[-1]
        else:
            break
    if  np.any(np.diff(np.array(dr)) < 0):
        total_time += 10000
    return total_time

def dd2(best_x, x):  #欧氏距离
    best_x = np.array(best_x)   #转化成numpy数组
    x = np.array(x)          #转化成numpy数组
    c = np.sum(pow(x - best_x, 2), axis=1)    #求方差，在行上的标准差
    d = pow(c, 0.5)   #标准差
    return d
def new_min(arr):  #求最小
    min_data = min(arr)   #找到最小值
    key = np.argmin(arr)  #找到最小值的索引
    return min_data, key
def type_x(xx,type,n):  #变量范围约束
    for v in range(n):
        if type[v] == -1:
            xx[v] = np.maximum(sub[v], xx[v])
            xx[v] = np.minimum(up[v], xx[v])
        elif type[v] == 0:
            xx[v] = np.maximum(sub[v], int(xx[v]))
            xx[v] = np.minimum(up[v], int(xx[v]))
        else:
            xx[v] = int(xx[v]%2)
    return xx
def woa(sub,up,type,nums,det):
    n = len(sub)  # 自变量个数
    num = nums * n  # 种群大小
    x = np.zeros([num, n])  #生成保存解的矩阵

    f = np.zeros(num)   #生成保存值的矩阵
    for s in range(num):      #随机生成初始解
        for v in range(n):
            rand_data = np.random.uniform(0,1)
            x[s, v] = sub[v] + (up[v] - sub[v]) * rand_data
        x[s, :] = type_x(x[s, :],type,n)
        f[s] = fun(x[s, :])
    best_f, a = new_min(f)  # 记录历史最优值
    best_x = x[a, :]  # 记录历史最优解
    trace = np.array([deepcopy(best_f)]) #记录初始最优值,以便后期添加最优值画图
    ############################ 改进的鲸鱼算法 ################################
    xx = np.zeros([num, n])
    ff = np.zeros(num)
    Mc = (up - sub) * 0.1  # 猎物行动最大范围
    for ii in tqdm(range(det)):      #设置迭代次数，进入迭代过程
        # 猎物躲避,蒙特卡洛模拟，并选择最佳的点作为下一逃跑点 #########！！！创新点
        d = dd2(best_x, x)  #记录当前解与最优解的距离
        d.sort()  #从小到大排序,d[0]恒为0
        z = np.exp(-d[1] / np.mean(Mc))  # 猎物急躁系数
        z = max(z, 0.1)     #决定最终系数
        yx = []  #初始化存储函数值
        dx = []  #初始化存储解
        random_rand = random.random() #0-1的随机数
        for i in range(10):    #蒙特卡洛模拟的次数
            m = [random.choice([-1, 1]) for _ in range(n)] #随机的-1和1
            asd = best_x + Mc * z * ((det-ii )/det) * random_rand * m   #最优解更新公式
            xd = type_x(asd,type,n)  #对自变量进行限制
            if i < 1:
                dx = deepcopy(xd)
            else:
                dx = np.vstack((dx,xd))   #存储每一次的解
            yx=np.hstack((yx,fun(xd)))    #存储每一次的值
        best_t, a = new_min(yx)  # 选择最佳逃跑点
        best_c = dx[a, :]   #最佳逃跑点
        if best_t < best_f:   #与鲸鱼算法得到的最优值对比
            best_f = best_t   #更新最优值
            best_x = best_c   #更新最优解
        ############################# 鲸鱼追捕 #################################
        w = (ii / det)**3   #自适应惯性权重!!!创新点
        a = (2 - 2*ii/det)*(1- w)  #a随迭代次数从2非线性下降至0！！！创新点
        pp=0.7 if ii <= 0.5*det else 0.4
        for i in range(num):
            r1 = np.random.rand()  # r1为[0,1]之间的随机数
            r2 = np.random.rand()  # r2为[0,1]之间的随机数
            A = 2 * a * r1 - a
            C = 2 * r2
            b = 1     #螺旋形状系数
            l = np.random.uniform(-1,1)  #参数l
            p = np.random.rand()
            if p < pp:
                if abs(A) >= 1:
                    rand_leader = np.random.randint(0, num)
                    X_rand = x[rand_leader, :]
                    D_X_rand = abs(C * X_rand - x[i, :])
                    xx[i, :] = w*X_rand - A * D_X_rand
                    xx[i, :] = type_x(xx[i, :],type,n) #对自变量进行限制
                elif abs(A) < 1:
                    D_Leader = abs(C * best_x - x[i, :])
                    xx[i, :] = w*best_x - A * D_Leader
                    xx[i, :] = type_x(xx[i, :],type,n) #对自变量进行限制
            elif p >= pp:
                D = abs(best_x - x[i, :])
                xx[i, :] = D*np.exp(b*l)*np.cos(2*np.pi*l) + (1-w)*best_x   #完整的气泡网捕食公式
                xx[i, :] = type_x(xx[i, :],type,n) #对自变量进行限制
            ff[i] = fun(xx[i, :])
            if len(np.unique(ff[:i]))/(i+1) <= 0.1:     #limit阈值 + 随机差分变异！！！创新点
                xx[i,:] = (r1*(best_x-xx[i,:]) +
                           r2*(x[np.random.randint(0,num),:] - xx[i,:]))
                xx[i, :] = type_x(xx[i, :],type,n) #对自变量进行限制
                ff[i] = fun(xx[i, :])
        #将上一代种群与这一代种群以及最优种群结合，选取排名靠前的个体组成新的种群
        F = np.hstack((np.array([best_f]), f, ff))
        F, b = np.sort(F,axis=-1,kind='stable'), np.argsort(F)#按小到大排序,获得靠前的位置
        X = np.vstack(([best_x], x, xx))[b, :]
        f = F[:num]  #新种群的位置
        x = X[:num, :]  #新种群的位置
        best_f, a = new_min(f)  # 记录历史最优值
        best_x = x[a , :]  # 记录历史最优解
        trace = np.hstack((trace, [best_f]))
    return best_x,best_f,trace

len_towns_list = len(towns_list)
sub = np.zeros((4*len_towns_list))  # 自变量下限
up = np.array([len_towns_list-i-1 for i in range(len_towns_list)]+
              len_towns_list * [1]+
              [len_towns_list] * len_towns_list*2)  # 自变量上限
type = np.array(len_towns_list*[0] + len_towns_list*[1] + 2*len_towns_list*[0])    #-1是有理数，0是整数，1是0-1变量
best_x,best_f,trace = woa(sub,up,type,50,200)     #种群大小，迭代次数
#种群大小可以为自变量个数，迭代次数看情况
print('最优解为：')
print(best_x)
print('最优值为：')
print(float(best_f))
print(fun(best_x,1))

'''plt.title('鲸鱼算法')
plt.plot(range(1,len(trace)+1),trace, color='r')
plt.show()'''
old_time = [
    0.05, 1.92, 2.08, 1.2, 0.25, 0.37, 0.44, 3.15, 0.28, 0.32,
    0.36, 0.14, 0.13, 0.32, 0.81, 0.89, 0.58, 0.05, 1.08, 0.67,
    0.82, 0.04, 0.82, 0.59, 0.67, 0.37, 0.95, 0.47,
    0.03, 0.12, 0.79, 1.41, 0.09, 0.19, 1.95, 1.01, 1.68, 1.47,
    0.80, 2.93, 1.54, 2.20, 1.05, 2.42, 2.87, 3.48, 1.74, 3.69,
    1.15, 1.28, 1.79, 1.92, 2.70, 2.09, 0.97, 0.73,
    1.10, 0.93, 1.81, 2.35, 2.86, 1.30, 0.04, 1.28, 0.23, 0.65,
    1.53, 0.80, 0.02, 1.02, 0.79, 0.47, 0.31, 1.20, 1.59, 1.88,
    4.16, 4.41, 0.66, 4.58, 0.42, 4.81, 0.01
]
new_time = [
    0.05, 0.47, 1.55, 0.06, 0.25, 0.37, 0.22, 0.37, 0.36, 0.81,
    0.28, 0.14, 0.28, 0.5, 0.57, 0.49, 0.25, 0.05, 1.69, 0.71,
    1.48, 0.04, 0.24, 0.36, 0.39, 0.66, 0.08, 0.48,
    0.03, 0.15, 1.92, 1.06, 0.08, 0.42, 1.38, 0.66, 1.11, 0.90,
    0.41, 0.25, 0.88, 2.10, 0.47, 1.07, 0.24, 2.45, 1.64, 2.67,
    3.32, 0.44, 0.23, 0.37, 2.62, 0.06, 1.32, 0.44,
    0.44, 0.28, 0.05, 1.50, 0.24, 2.86, 0.04, 0.62, 0.23, 0.68,
    0.47, 0.83, 0.02, 0.94, 0.19, 0.35, 0.02, 1.11, 0.70, 0.18,
    1.35, 0.07, 1.25, 1.06, 0.25, 0.02, 0.01
]