
#############################################################
g = float(input())
if g > 100:
    print('错误成绩')
else:
    g //= 10
    if g == 9 or g == 10:
        print('优秀')
    elif g == 8:
        print('良')
    elif g == 7:
        print('中')
    elif g == 6:
        print('及格')
    elif g >= 0 and g < 6:
        print('不及格')
    else:
        print('错误成绩')

####################################################################
def fenji(g):
    g = float(g)
    if g > 100:
        print('错误成绩')
    else:
        g //= 10
        if g == 9 or g == 10:
            print('优秀')
        elif g == 8:
            print('良')
        elif g == 7:
            print('中')
        elif g == 6:
            print('及格')
        elif g >= 0 and g < 6:
            print('不及格')
        else:
            print('错误成绩')
shuru = list(eval(input('成绩之间用英文逗号隔开：')))
for i in range(len(shuru)):
    print(f'成绩{shuru[i]}:',end='')
    fenji(shuru[i])


import math
r = float(input())
if r <= 0:
    print('请输入大于0的整数或实数')
else:
    print(round(math.pi*r**2,2))


for i in range(1,10):
    print()
    for v in range(1,i+1):
        print(f'{v}x{i}={i*v}',end=' ')

print('\n')

a = 0
b = 0
while True:
    h = eval(input('请输入一个数：'))
    if h == 0:
        break
    elif h > 0:
        a += 1
    elif h < 0:
        b +=1
print(f'正数个数为{a},负数个数为{b}')

