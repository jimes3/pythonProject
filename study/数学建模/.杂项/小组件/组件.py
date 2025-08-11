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


'''
print("最优适应度为：", format(best_fitness,'.6f'))
print(f"最优适应度为：{best_fitness:.6f}")  #保留六位小数
###################################################################################################
数据读取:
data = pd.read_excel('path', sheetname = 'sheet1', header = 0, names = ['第一列','第二列','第三列'])
    path：要读取的文件绝对路径
    sheetname：指定读取excel中哪一个工作表，默认sheetname = 0，即默认读取excel中的第一个工作表
        若sheetname = ‘sheet1’，即读取excel中的sheet1工作表；
        若sheetname = ‘汇总’，即读取excel中命名为“汇总”的工作表；
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
plt:
    fig:画布    ax:图表
    # 创建画布,只是一个表可以不创建
    plt.figure(figsize=(9, 6))
    fig,ax = plt.subplots(figsize=(9,6))
    #改变图的比例
    plt.rcParams["figure.figsize"] = (4, 8)
    # 子网络
    plt.subplot(2, 3, i+1)
    
    # 画出原始值的曲线
    plt.plot(x, y, color='k', label='y')
    # 画出各个模型的预测线
    plt.plot(x, pre_y, color_list[i], label=model_names[i])
    #散点图
    ax.scatter(x,y,(z),marker='x',s=50,c='black')
    #网格图
    plt.grid(color='blue', linestyle='-.', linewidth=0.7)
    
    #坐标轴
    plt.xlabel('k')
    plt.ylabel('轮廓系数')
    plt.xlim(11, 17)
    plt.ylim(9, 16)
    #设置坐标轴刻度
    my_x_ticks = np.arange(-5, 5, 0.5)
    #对比范围和名称的区别
    #my_x_ticks = np.arange(-5, 2, 0.5)
    my_y_ticks = np.arange(-2, 2, 0.3)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    #刻度倾斜
    plt.xticks (rotation =30)
    #设置文字刻度
    plt.yticks([-2, -1.8, -1, 1.22, 3],[r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
    
    plt.title(model_names[i])
    plt.legend(loc='lower left')   #标注在左下角
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    
    #设置坐标轴–边框
    gca():获取当前坐标轴信息
    .spines:设置边框
    .set_color:设置边框颜色：默认白色
    .spines:设置边框
    .xaxis.set_ticks_position:设置x坐标刻度数字或名称的位置
    .yaxis.set_ticks_position:设置y坐标刻度数字或名称的位置
    .set_position:设置边框位置
    
    #设置上边和右边无边框
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    #设置x坐标刻度数字或名称的位置
    ax.xaxis.set_ticks_position('bottom')
    #设置边框位置
    ax.spines['bottom'].set_position(('data', 0))

    
    #定义图像和三维格式坐标轴
    fig=plt.figure()
    ax1 = Axes3D(fig)
    
    ax1.scatter3D(xd,yd,zd, cmap='Blues')  #绘制散点图
    ax1.plot3D(x,y,z,'gray')    #绘制空间曲线
    ax1.plot_surface(X,Y,Z,cmap='rainbow')  #绘制空间曲面
    #ax1.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
    
    #保存
    plt.savefig(f'{z}现实对比.png',dpi=600)
    plt.show()
'''