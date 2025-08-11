import random
def cal_pai_mc(n=1000000):
    r = 1.0
    a, b = (0.0, 0.0)
    x_neg, x_pos = a - r, a + r
    y_neg, y_pos = b - r, b + r
    m = 0
    for i in range(0,n):
        x = random.uniform(x_neg, x_pos)
        y = random.uniform(y_neg, y_pos)
        if x**2 + y**2 <= 1.0:
            m += 1
    h = (m / n) * 4
    print(format(h, '.6f'))
cal_pai_mc(n=10000000)
