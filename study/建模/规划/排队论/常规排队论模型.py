import math

def mm1_model(lamb_da, miu, n=0):
    rou = lamb_da / miu
    W_s = 1 / (miu - lamb_da)
    W_q = rou / (miu - lamb_da)
    L_s = lamb_da * W_s
    L_q = lamb_da * W_q
    P_n = (1-rou)*(rou**n)
    return {
        '服务强度': rou,
        '平均队长': L_s,
        '平均队列长': L_q,
        '平均逗留时间': W_s,
        '平均等待时间': W_q,
        f'系统稳定有{n}个顾客的概率':P_n
    }
def mm1N_model(lamb_da, miu, N, n=0):
    '''
    损失制:当顾客到达时，队列达到N，顾客随即离去。
    '''
    rou = lamb_da / miu
    P_N = (1-rou)*(rou**N)/(1-rou**(N+1))
    lamb_da_e = lamb_da*(1-P_N)
    rou_e = lamb_da_e / miu
    L_s = 1/(1-rou) - ((N+1)*rou**(N+1))/(1-rou**(N+1))
    L_q = L_s - rou_e
    W_s = L_s/lamb_da_e
    W_q = L_q/lamb_da_e
    P_n = (1-rou)*(rou**n)/(1-rou**(N+1))
    return {
        '有效服务强度': rou_e,
        '平均队长': L_s,
        '平均队列长': L_q,
        '平均逗留时间': W_s,
        '平均等待时间': W_q,
        f'系统稳定有{n}个顾客的概率':P_n
    }
def mm1_m_model(lamb_da, miu, m, n=0):
    '''
    这里的lamb_da定义为每个顾客的到达率
    '''
    rou = lamb_da / miu
    P_0 = 1/sum(math.factorial(m)/math.factorial(m-i)*rou**i
                for i in range(0,m+1))
    W_s = m/(miu*(1-P_0)) - 1/lamb_da
    W_q = W_s - 1/miu
    L_s = m - (miu/lamb_da)*(1-P_0)
    L_q = L_s - (1-P_0)
    if n == 0:
        P_n = P_0
    else:
        P_n = math.factorial(m)/math.factorial(m-n)*rou**n * P_0
    return {
        '服务强度': rou,
        '平均队长': L_s,
        '平均队列长': L_q,
        '平均逗留时间': W_s,
        '平均等待时间': W_q,
        f'系统稳定有{n}个顾客的概率':P_n
    }
def mmc_model(lamb_da, miu, c, n=0):
    rou = lamb_da / (miu*c)
    P_0 = 1/(sum(1/math.factorial(i)*(rou*c)**i
                for i in range(0,c-1))
             +1/(math.factorial(c)*(1-rou))*(rou*c)**c)
    L_q = P_0*(rou*c)**c*rou/(math.factorial(c)*(1+rou)**2)
    L_s = L_q + rou*c
    W_s = L_s / lamb_da
    W_q = L_q / lamb_da
    if n == 0:
        P_n = P_0
    elif n<=c:
        P_n = P_0*(rou*c)**n / math.factorial(n)
    else:
        P_n = P_0*(rou*c)**n / (math.factorial(c)*c**(n-c))
    return {
        '服务强度': rou,
        '平均队长': L_s,
        '平均队列长': L_q,
        '平均逗留时间': W_s,
        '平均等待时间': W_q,
        f'系统稳定有{n}个顾客的概率':P_n
    }
def mmcN_model(lamb_da, miu, c, N, n=0):
    rou = lamb_da / (miu*c)
    if rou == 1:
        P_0 = sum(c**i/math.factorial(i) for i in range(0,c))\
             +(N-c+1)*c**c/math.factorial(c)
    else:
        P_0 =1/(sum(c*rou**i/math.factorial(i) for i in range(0,c))\
                +(rou*(rou**c-rou**N)/(1-rou))*c**c/math.factorial(c))
    if n<=c:
        P_n = P_0*(rou*c)**n / math.factorial(n)
    elif n>c and n<=N:
        P_n = P_0 * c**c * rou**n / math.factorial(c)
    L_q = 0
    P_N = P_0 * c**c * rou**N / math.factorial(c)
    for i in range(1,N-c+1):
        L_q += i*P_N
    L_s = L_q + rou*c*(1-P_N)
    W_q = L_q / (lamb_da*(1-P_N))
    W_s = W_q + 1/miu
    return {
        '服务强度': rou,
        '平均队长': L_s,
        '平均队列长': L_q,
        '平均逗留时间': W_s,
        '平均等待时间': W_q,
        f'系统稳定有{n}个顾客的概率': P_n
    }
def mmc_m_model(lamb_da, miu, c, m, n=0):
    '''
    n应该要大于c，不然概率可能大于1.
    因为有时每个服务台都会有人。
    '''
    rou = m*lamb_da / (miu*c)
    P_0 =1/((sum((c*rou/m)**i/(math.factorial(i)*math.factorial(m-i)) for i in range(0,c+1))\
            +sum(c**c*(rou/m)**i/(math.factorial(m-i)*math.factorial(c)) for i in range(0,c+1)))\
            *math.factorial(m))
    def P(P_0, lamb_da, miu, c, m, n=0):
        if n<=c:
            P_n = (P_0*((lamb_da/miu)**n)*math.factorial(m) /
                   (math.factorial(n)*math.factorial(m-n)))
        elif n>c and n<=m:
            P_n = (P_0*(lamb_da/miu)**n*math.factorial(m) /
                   (math.factorial(c)*math.factorial(m-n)*c**(n-c)))
        return P_n
    P_n = P(P_0, lamb_da, miu, c, m, n)
    L_q = sum(i*P(P_0, lamb_da, miu, c, m, i) for i in range(1,m-c+1))
    L_s = sum(i*P(P_0, lamb_da, miu, c, m, i) for i in range(1,m+1))
    lamb_da_e = lamb_da * (m-L_s)
    W_s = L_s/lamb_da_e
    W_q = L_q/lamb_da_e
    return {
        '服务强度': rou,
        '平均队长': L_s,
        '平均队列长': L_q,
        '平均逗留时间': W_s,
        '平均等待时间': W_q,
        f'系统稳定有{n}个顾客的概率': P_n
    }
# 输入模拟参数
lamb_da = 0.8  # 平均到达速率
miu = 0.6  # 平均服务速率
c = 10  # 服务台数量 ,需要小于m
N = 5  # 队列（排队）容量
m = 10  # 顾客源总数

results = mmc_m_model(lamb_da, miu, c, m, 5)
for key, value in results.items():
    print(f"{key}: {value:.6f}")
