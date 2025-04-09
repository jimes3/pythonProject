import pandas as pd

#############第一小问
df = pd.read_excel("D:\.jimes\下载\C题\附件.xlsx",sheet_name = '表单1')
df = df.dropna()
def func(x):
    if x == 'A':
        x = 1
    elif x == 'B':
        x = 2
    elif x == 'C':
        x = 3
    elif x == '无风化':
        x = 0
    elif x == '风化':
        x = 1
    elif x == '高钾':
        x = 0
    elif x == '铅钡':
        x = 1
    elif x == '浅绿':
        x = 1
    elif x == '绿':
        x = 2
    elif x == '深绿':
        x = 3
    elif x == '蓝绿':
        x = 4
    elif x == '浅蓝':
        x = 5
    elif x == '深蓝':
        x = 6
    elif x == '紫':
        x = 7
    elif x == '黑':
        x = 8
    return x
df['纹饰'] = df['纹饰'].apply(func)
df['类型'] = df['类型'].apply(func)
df['颜色'] = df['颜色'].apply(func)
df['表面风化'] = df['表面风化'].apply(func)

'''
# 创建一个2x2的观察频率矩阵
observed = np.array([[10, 15], [20, 25]])
# 进行卡方检验
chi2, p, dof, expected = chi2_contingency(observed)
# 输出结果
print("卡方统计量:", chi2)
print("p值:", p)
print("自由度:", dof)
print("期望频率:", expected)
'''