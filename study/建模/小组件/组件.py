# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#plt.ion()#把matplotlib改为交互状态
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文

for i in range(9):     #批量生成变量名
    locals()['v'+ str(i)] = i
print(v1)
print(locals()['v1'])
'''
print("最优适应度为：", format(best_fitness,'.6f'))
print(f"最优适应度为：{best_fitness:.6f}")  #保留六位小数
df.to_excel('output.xlsx', index=False)
###################################################################################################
数据读取:
data = pd.read_excel('path', sheet_name = 'Sheet1', header = 0, names = ['第一列','第二列','第三列'])
    path：要读取的文件绝对路径
    sheet_name：指定读取excel中哪一个工作表，默认sheet_name = 0，即默认读取excel中的第一个工作表
        若sheet_name = ‘sheet1’，即读取excel中的sheet1工作表；
        若sheet_name = ‘汇总’，即读取excel中命名为“汇总”的工作表；
    header：用作列名的行号，默认为header = 0
        若header = None，则表明数据中没有列名行；
        若header = 0，则表明第一行为列名；
    names：列名命名或重命名
data = pd.read_csv('path',sep = ',', header = 0, names = ['第一列','第二列','第三列'], encoding = 'utf-8')
    path：要读取的文件绝对路径
    sep：指定列与列间的分隔符，默认sep = ‘,’
        若sep = ‘\t’，即列与列间用制表符\t分隔；
        若sep = ‘,’，即列与列间用逗号,分隔；
    header：用作列名的行号，默认为0
        若header = None，则表明数据中没有列名行；
        若header = 0，则表明第一行为列名；
    names：列名命名或重命名
    encoding：指定用于unicode文本编码格式
        若encoding = ‘utf-8’，则表明用UTF-8编码的文本；
        若encoding = ‘gbk’，则表明用gbk编码的文本；
data = pd.read_table('path', sep = '\t', header = None, names = ['第一列','第二列','第三列'])
    path：要读取的文件绝对路径
    sep：指定列与列间的分隔符，默认sep = ‘\t’
        若sep = ‘\t’，即列与列间用制表符\t分隔；
        若sep = ‘,’，即列与列间用逗号,分隔；
    header：用作列名的行号，默认为header = 0
        若header = None，则表明数据中没有列名行；
        若header = 0，则表明第一行为列名；
    names：列名命名或重命名
#######################################################################################
panda:
    # series的索引
    index = ['2019/3/23', '2019/3/24', '2019/3/25', '2019/3/26', '2019/3/27',
         '2019/3/28', '2019/3/29', '2019/3/30', '2019/3/31', '2019/4/1',
         '2019/4/2', '2019/4/3', '2019/4/4']
    # series的值
    value = [18, 20, 19, 20, 18, 15, 17, 19, 20, 15, 18, 15, 20]
    # 创建series
    # 如果不指定索引，会自动生成0,1,2,....这样的索引
    s = pd.Series(data=value, index=index)
    print(s)
    # 输出
    # 2019/3/23    18
    # 2019/3/24    20
    # 2019/3/25    19
    。。。。。
    
    # apply
    # 对姓名这一列的每个元素添加字母'xm'
    def myfunc(x):
        return 'xm' + x
    df['姓名'] = df['姓名'].apply(myfunc)
    # 输出
    #    学号  班级    姓名 性别   身高  语文成绩   学分         日期
    # 0  x1  1班  xm张三  男  177    92  1.5  2019/3/23
    # 1  x2  1班  xm李四  男  151    84  2.3  2019/3/24
    # ......
    # 对成绩这一列，如果成绩小于90分则改成90份
    def myfunc(x):
        if x < 90:
            return 90
        else:
            return x
    df['语文成绩'] = df['语文成绩'].apply(myfunc)
    # 输出
    #    学号  班级  姓名 性别   身高  语文成绩   学分         日期
    # 0  x1  1班  张三  男  177    92  1.5  2019/3/23
    # 1  x2  1班  李四  男  151    90  2.3  2019/3/24
    # ......
    # 分组应用apply
    # 需要注意的是，myfunc接收的参数是pandas.Series类型
    def myfunc(series):
        return series.max()
    # 计算每个班级语文成绩最高分
    df.groupby(by=['班级'], as_index=False)['语文成绩'].apply(myfunc)
    # 0    92
    # 1    90
######################################################################################
numpy:
         ###用惯了dataframe，用numpy格式的可以简单理解为行列互换，原来的行变列，列变行####
    #基本原理就是把数字都排成一行，再来分
    print('数组的维度是：', a.shape)
    print('将二维数组转成一维数组', a.ravel()) # 输出 将二维数组转成一维数组 [1 2 3 4 5 6]
    print('改变二维数组形状：2*3 -> 3*2 \n', a.reshape((3,2)))
    print('将二维数组转成列向量：\n', a.reshape((-1,1)))
    c = np.hstack((a,b)) #行上的拼接
    c = np.vstack((a,b)) #列上的拼接
    # 将两个二维矩阵沿着第三个维度（axis=2）堆叠起来，形成一个三维矩阵
    result = np.stack((yun23, dp), axis=2)
#####################################################################################
'''