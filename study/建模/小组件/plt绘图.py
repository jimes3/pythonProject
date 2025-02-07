import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文

x = np.arange(0,5,0.1)
y = np.sin(x)

# 创建画布
fig,ax = plt.subplots(figsize=(9,6))
############################################################
plt.subplot(2, 2, 1)
# 折线图
plt.plot(x,y,color='C29', label='y',ls='--')
# 坐标轴
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 5)
plt.ylim(-1, 1)
# 设置坐标轴刻度
plt.xticks(np.arange(0, 6, 1))
plt.yticks(np.arange(-1, 1, 0.5))
# 刻度倾斜
plt.xticks (rotation =0)
# 设置文字刻度
plt.yticks([-1,-0.5,0,0.5,1],[r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'],
           rotation =0)
plt.title('title')
plt.legend(loc='upper right')
#plt.legend(bbox_to_anchor=(1.1, 1.05))
#############################################################
plt.subplot(2, 2, 2)
# 散点图
plt.scatter(x,y,color='C29', label='y',s=5)
###############################################
X = [[1.2, 2.3, 3.0, 4.5],
     [1.1, 2.2, 2.9, 5.0]]
df = pd.DataFrame(X, index=['Apple', 'Orange'])
# 箱线图
df.T.boxplot(showmeans=True,boxprops={'linewidth': 1})
############################################################
plt.subplot(2, 2, 3)
# 饼状图
plt.pie([3,4,5], labels=['第一类','第二类','第三类'], autopct='%1.3f%%', startangle=90)
############################################################
ax = plt.subplot(2, 2, 4,polar=True)
# 隐藏网格线
#ax.grid(False)
# 隐藏刻度值
ax.set_yticklabels([])
# 设置 r 轴范围
ax.set_rlim(0, 65)
# 设置 r 轴刻度
ax.set_rticks([0, 20, 40, 60])
# 极坐标图
employee = ["Sam", "Rony", "Albert", "Chris", "Jahrum"]
# 首末一样，为了连线
actual = [45, 53, 55, 61, 57, 45]
expected = [50, 55, 60, 65, 55, 50]
theta = np.linspace(0, 2 * np.pi, len(actual))
# 按度数添加注释
lines, labels = plt.thetagrids(range(0, 360, int(360/len(employee))), (employee))
# 实际图
plt.scatter(theta, actual, c=['b','c','g','k','m','r'], marker='o',s=10)
# 期待图
plt.plot(theta, expected, lw=0.9)
# 两点间直线
plt.plot([np.pi,0.5*np.pi],[50,60])
# 填充
plt.fill(theta, actual, 'b', alpha=0.08)
plt.legend(labels=('Actual', 'Expected','实际填充'), bbox_to_anchor=(1.2, 1.05))#loc='upper right')
plt.title("极坐标系图")

plt.show()

'''
fig:画布    ax:图表
    
    #设置坐标轴–边框
    ax = plt.gca():获取当前坐标轴信息
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
    ax1.scatter3D(x,x,x, cmap='Blues')  #绘制散点图
    ax1.plot3D(x,y,x,'gray')    #绘制空间曲线
    #ax1.plot_surface(x,x,x,cmap='rainbow')  #绘制空间曲面
    #ax1.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
    
    #保存
    #plt.savefig(f'{z}现实对比.png',dpi=600)
    plt.show()
'''