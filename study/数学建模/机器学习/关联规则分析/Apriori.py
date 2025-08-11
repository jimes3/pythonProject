import numpy as np
import pandas as pd

# 测试数据集，用于debug
def loadDataSet():
    return [['a', 'c', 'e'], ['b', 'd'], ['b', 'c'], ['a', 'b', 'c', 'd'], ['a', 'b'], ['b', 'c'], ['a', 'b'],
            ['a', 'b', 'c', 'e'], ['a', 'b', 'c'], ['a', 'c', 'e']]


def loaddata():
    order_data = pd.read_csv('GoodsOrder.csv', header=0, encoding='gbk')
    # 转换数据格式
    order_data['Goods'] = order_data['Goods'].apply(lambda x: "," + x)
    order_data = order_data.groupby('id').sum().reset_index()
    order_data['Goods'] = order_data['Goods'].apply(lambda x: [x[1:]])
    order_data_list = list(order_data['Goods'])
    # print(order_data_list)
    # 分割商品名为每一个元素
    data_translation = []
    for i in order_data_list:
        # 列表中元素为1个字符串，所以用0索引
        p = i[0].split(",")
        data_translation.append(p)
    return data_translation


def creatC1(data):
    """
    找到1项候选集C1
    :param data: 数据集
    :return: 1项候选集C1
    """
    C1 = []
    for row in dataSet:
        for item in row:
            if [item] not in C1:
                C1.append([item])
    # 中文字符串升序排序
    C1.sort()
    # frozenset()返回一个冻结的集合，冻结后集合不能再添加或删除任何元素
    return list(map(frozenset, C1))


def calSupport(D, C, minSupport):
    """
    计算1项候选集的支持度,剔除小于最小支持度的项集，
    :param D: 数据集
    :param C1: 候选集
    :param minSupport: 最小支持度
    :return: 返回1项频繁集及其支持度
    """
    dict_sup = {}  # 中间储存变量，用于计数
    # 迭代每一条数据，对项集中的每一项进行计数
    for i in D:
        for j in C:
            # 集合j是否是集合i的子集，如果是返回True，否则返回False
            if j.issubset(i):
                # 再判断之前有没有统计过，没有统计过的话为1
                if j not in dict_sup:
                    dict_sup[j] = 1
                else:
                    dict_sup[j] += 1
    # 事务总数
    sumCount = float(len(D))
    # 计算支持度，支持度 = 项集的计数/事务总数
    supportData = {}  # 用于存储频繁集的支持度
    relist = []  # 用于存储频繁集
    for i in dict_sup:
        temp_sup = dict_sup[i] / sumCount
        # 将剔除后的频繁项集及其对应支持度保存起来
        if temp_sup > minSupport:
            relist.append(i)
            supportData[i] = temp_sup
    # 返回1项频繁项集及其对应支持度
    return relist, supportData


def aprioriGen(Lk, k):
    """
    改良了剪枝步，原来的Ck是由L1与L(k-1)来连接产生的，这里采用了新的连接方式
    使用剪枝算法，减少了候选集空间，找到k项候选集
    :param Lk: k-1项频繁集
    :param k: 第k项
    :return: 第k项候选集
    """
    reList = []  # 用来存储第k项候选集
    lenLk = len(Lk)  # 第k-1项频繁集的长度
    # 两两组合遍历
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            # 前k-1项相等，则可相乘，这样可以防止重复项出现
            if L1 == L2:
                a = Lk[i] | Lk[j]  # a为frozenset集合
                # 进行剪枝
                a1 = list(a)  # a1为k项集中的一个元素
                b = []  # b为它的所有k-1项子集
                # 构造b:遍历取出每一个元素，转换为set，依次从a1中剔除该元素，并加入到b中
                for q in range(len(a1)):
                    t = [a1[q]]
                    tt = frozenset(set(a1) - set(t))
                    b.append(tt)

                # 当b都是频繁集时，则保留a1,否则，删除
                t = 0
                for w in b:
                    # 如果为True，说明是属于候选集，否则不属于候选集
                    if w in Lk:
                        t += 1
                # 如果它的子集都为频繁集，则a1是候选集
                if len(b) == t:
                    reList.append(b[0] | b[1])

    return reList


def scanD(D, Ck, minSupport):
    """
    计算候选k项集的支持度，剔除小于最小支持度的候选集，得到频繁k项集及其支持度
    :param D: 数据集
    :param Ck: 候选k项集
    :param minSupport: 最小支持度
    :return: 返回频繁k项集及其支持度
    """
    sscnt = {}  # 存储支持度
    for tid in D:  # 遍历数据集
        for can in Ck:  # 遍历候选项
            if can.issubset(tid):  # 判断数据集中是否含有候选集各项
                if can not in sscnt:
                    sscnt[can] = 1
                else:
                    sscnt[can] += 1

    # 计算支持度
    numItem = len(D)  # 事务总数
    reList = []  # 存储k项频繁集
    supportData = {}  # 存储频繁集对应支持度
    for key in sscnt:
        support = sscnt[key] / numItem
        if support > minSupport:
            reList.insert(0, key)  # 满足条件的加入Lk中
            supportData[key] = support
    return reList, supportData


def apriori(dataSet, minSupport=0.2):
    """
    apriori关联规则算法
    :param data: 数据集
    :param minSupport: 最小支持度
    :return: 返回频繁集及对应的支持度
    """
    # 首先，找到1项候选集
    C1 = creatC1(dataSet)
    # 使用list()转化为列表，用于支持度计算
    D = list(map(set, dataSet))
    # 计算1项候选集的支持度,剔除小于最小支持度的项集，返回1项频繁集及其支持度
    L1, supportData = calSupport(D, C1, minSupport)
    L = [L1]  # 加列表框，使得1项频繁集称为一个单独的元素

    k = 2  # k项
    # 跳出循环的条件是没有候选集
    while len(L[k - 2]) > 0:
        # 产生k项候选集Ck
        Ck = aprioriGen(L[k - 2], k)
        # 计算候选k项集的支持度，剔除小于最小支持度的候选集，得到频繁k项集及其支持度
        Lk, supK = scanD(D, Ck, minSupport)
        # 将supK中的键值对添加到supportData
        supportData.update(supK)
        # 将第k项的频繁集添加到L中
        L.append(Lk)  # L的最后一个值为空值
        k += 1
    del L[-1]
    # 返回频繁集及其对应的支持度；L为频繁项集，是一个列表，1,2，3项集分别为一个元素
    return L, supportData


def getSubset(fromList, totalList):
    """
    生成集合的所有子集
    :param fromList:
    :param totalList:
    """
    for i in range(len(fromList)):
        t = [fromList[i]]
        tt = frozenset(set(fromList) - set(t))  # k-1项子集

        if tt not in totalList:
            totalList.append(tt)
            tt = list(tt)
            if len(tt) > 1:
                getSubset(tt, totalList)  # 所有非1项子集


def calcConf(freqSet, H, supportData, ruleList, minConf):
    """
    计算置信度，并剔除小于最小置信度的数据,这里利用了提升度概念
    :param freqSet: k项频繁集
    :param H: k项频繁集对应的所有子集
    :param supportData: 支持度
    :param RuleList: 强关联规则
    :param minConf: 最小置信度
    """
    # 遍历freqSet中的所有子集并计算置信度
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]  # 相当于把事务总数抵消了

        # 提升度lift计算lift=p(a&b)/p(a)*p(b)
        lift = supportData[freqSet] / (supportData[conseq] * supportData[freqSet - conseq])
        if conf >= minConf and lift > 1:
            print(freqSet - conseq, '-->', conseq, '支持度', round(supportData[freqSet], 6), '置信度：', round(conf, 6),
                  'lift值为：', round(lift, 6))
            ruleList.append((freqSet - conseq, conseq, conf))


def get_rule(L, supportData, minConf=0.7):
    """
    生成强关联规则：频繁项集中满足最小置信度阈值，就会生成强关联规则
    :param L: 频繁集
    :param supportData: 支持度
    :param minConf: 最小置信度
    :return: 返回强关联规则
    """
    bigRuleList = []  # 存储强关联规则
    # 从2项频繁集开始计算置信度
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = list(freqSet)
            all_subset = []  # 存储H1的所有子集
            # 生成所有子集
            getSubset(H1, all_subset)
            # print(all_subset)
            # 计算置信度，并剔除小于最小置信度的数据
            calcConf(freqSet, all_subset, supportData, bigRuleList, minConf)
    return bigRuleList


if __name__ == '__main__':
    dataSet = loaddata()
    # 返回频繁集及其对应的支持度
    L, supportData = apriori(dataSet, minSupport=0.02)
    # 生成强关联规则
    rule = get_rule(L, supportData, minConf=0.35)