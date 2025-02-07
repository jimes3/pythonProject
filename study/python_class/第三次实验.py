def fun1(a,b,c):
    return max(a,b,c)

def fun2(x):
    list = []
    for i in range(2,x):
        list.append(x % i)
    if all(list) != 0:
        print('是素数')
    else:
        print('不是素数')
fun2(3)

def fun3(a,b):
    n = a*b
    while b != 0:
        a,b = b,a%b
    return a,n/a

a,b = fun3(5,8)
print(a,b)
