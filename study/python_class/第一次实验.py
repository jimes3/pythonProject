
sd = input('写出三个数相乘,乘号为*：')
df = sd.split('*')
print(f'{df[0]}*{df[1]}*{df[2]}={int(df[0])*int(df[1])}')

s = 100
print('二进制转换：',bin(s))
print('八进制转换：',oct(s))
print('十六进制转换：',hex(s))
print("%e" % 123.45)

str1 = '南京'
str2 = '江苏'
print(f'{str1}是{str2}省会')
print('{}是{}省会'.format(str1,str2))
print('{0}是{1}省会'.format(str1,str2))
print(str1+'是'+str2+'省会')

def ceil(x):
    df = str(x).split('.')
    return float(df[0])+1
re = ceil(4.56)
print(re)