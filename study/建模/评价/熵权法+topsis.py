import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出

#导入数据
data = pd.read_excel('收发货表格.xlsx',usecols=['发货','收货','总运输','发斜','收斜','平均相关性','发货城市','收货城市','连接数'])
x_data = np.array(data)

#正向化
def positive(x, type, best_value=None, a=None, b=None):
    '''
    :param x: 原始数据
    :param type: 1表示极小型，2表示中间型，3表示区间型,4表示正数话，[[,1],[,2],[,3]]前面是需要正向化的列的序号
    :param best_value: 中间型的最优值
    :param a: 区间型的区间下限
    :param b: 区间型的区间上限
    :return: 正向化后的数据（列）
    '''
    if type == None:     #先判断是否需要正向化
        pass
    else:
        x = x.T  #转置
        m = np.array(type).shape[0]  #获得需要正向化的列数
        for i in range(int(m)):  #迭代需要正向化的列
            if type[i][1] == 1:   #成本型数据的转化，采用max-x
                x[type[i][0]] = x[type[i][0]].max(0)-x[type[i][0]]
            elif type[i][1] == 2:  #中间型数据的转化
                max_value = (abs(x[type[i][0]] - best_value)).max()
                x[type[i][0]] = 1 - abs(x[type[i][0]] - best_value) / max_value
            elif type[i][1] == 3:  #区间型数据的转化
                max_value = (np.append(a-x[type[i][0]].min(),x[type[i][0]].max()-b)).max()  #即M，后面转换时的分母
                x_rows = x[type[i][0]].shape[0]
                for v in range(x_rows):
                    if x[type[i][0]][v] > b:  #当数据大于区间最大值的转换
                        x[type[i][0]][v] = 1-(x[type[i][0]][v]-b)/max_value
                    elif x[type[i][0]][v] < a:   #当数据小于区间最小值的转换
                        x[type[i][0]][v] = 1-(a-x[type[i][0]][v])/max_value
                    elif a <= x[type[i][0]][v] <= b:  #当数据在区间内则给予最优值1
                        x[type[i][0]][v] = 1
                    else:                      #其它情况则给予最劣值0
                        x[type[i][0]][v] = 0
            elif type[i][1] == 4:   #极大型负数的转换
                x[type[i][0]] = 1-(x[type[i][0]].min(0)) + x[type[i][0]]
        return x.T

#标准化
def normalize(x):
    sqrt_sum = (x * x).sum(axis=0)  #每一列元素的平方和
    sqt_sum_z = np.tile(sqrt_sum, (x.shape[0], 1)) #转换格式,每一列的值均为当列的元素平方和
    Z = x / np.sqrt(sqt_sum_z) #标准化
    return Z

#熵权法计算权值
def importance(data):
    l = len(data[0])
    h = [0]*l
    w = [0]*l
    data = data.T
    for v in range(l):
        for i in range(len(data[0])):
            if data[v][i] == 0:
                pass
            else:
                h[v] += -data[v][i] * np.log(data[v][i])/np.log(len(data[0]))       #计算每个指标的信息熵值
    for i in range(l):
        w[i] = (1-h[i])/(len(data)-sum(h))           #计算每个指标的权重
    return w
#topsis算法
def topsis(z,h):
    z_max = z.max(0)  #每一列的最大值与最小值
    z_min = z.min(0)

    d_m = z - np.tile(z_max, (z.shape[0], 1))  #每个方案与最优解及最劣解的差值数列
    d_i = z - np.tile(z_min, (z.shape[0], 1))

    d_i_max = np.sqrt(((h * d_m) ** 2).sum(axis=1))  #每个方案的综合距离
    d_i_min = np.sqrt(((h * d_i) ** 2).sum(axis=1))

    score = d_i_min/(d_i_max + d_i_min) #每个方案的评分
    std_score = score / score.sum(axis=0)  #归一化，方便查看
    return std_score

if __name__ == "__main__":
    #正向化，如果不需要则为空列表即可,要求都为正向化且数据都大于0
    type = [[3,4],[4,4],[1,4],[7,4]]
    a = positive(x_data,type,best_value=None, a=None, b=None)
    #标准化
    b = normalize(a)
    #信息熵计算
    h = importance(b)
    #topsis评价
    s = topsis(b,h)
    #将评价与方案连接，方便对比与观察
    clo = np.array(['A','B','C','D','E','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y'])
    lianjie_list = []
    for i in range(len(s)):
        lianjie_list.append([clo[i], s[i]])
    jieguo = sorted(lianjie_list,reverse=True, key=lambda x: x[1])  #根据评分值进行排序
    print(jieguo)
