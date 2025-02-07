import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文
'''
# 读取Excel文件
df = pd.read_excel('point.xlsx', sheet_name = 'Sheet1', header = None, names = ['县','乡镇','坐标','人口','纬度','经度'])
for i in range(83):
    df.iat[i,4]=eval(df.iat[i,2].split(',')[0])
    df.iat[i,5]=eval(df.iat[i,2].split(',')[1])
print(df)
df.to_excel('output.xlsx', index=False)
'''
df = pd.read_excel('output.xlsx', sheet_name = 'Sheet1', header = 0)
print(df)

#plt.scatter(df['经度'], df['纬度'])
for i, label in enumerate(df['乡镇']):
    if 10 <= i <= 19:
        plt.scatter(df['经度'][i], df['纬度'][i],color='black')
        plt.text(df['经度'][i], df['纬度'][i], label, fontsize=7, ha='right', va='bottom')#, rotation=180)
    elif i>=83:
        plt.scatter(df['经度'][i], df['纬度'][i],color='red')
        plt.text(df['经度'][i], df['纬度'][i], label, fontsize=7, ha='right', va='bottom', color='red')#,rotation=180)
plt.show()