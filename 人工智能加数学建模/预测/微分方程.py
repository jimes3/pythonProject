import numpy as np
from scipy.integrate import odeint,solve_ivp
import matplotlib.pyplot as plt
import sympy
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

#################################      python 解析解     ###########################################
## 初始化打印环境
sympy.init_printing()
## 标记参数且均为正
t, omega0, gamma = sympy.symbols("t, omega_0, gamma", positive=True)
## 标记x是微分函数，非变量
x = sympy.Function('x')

## 通过diff建立函数方程#####################################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ode = x(t).diff(t, 2) + 2 * gamma * omega0 * x(t).diff(t) + omega0**2*x(t)
## 将初始条件字典匹配
ics = {x(0): 1, x(t).diff(t).subs(t, 0): 0}  #subs(t, 0)表示t为0
## 通过dsolve得到通解
ode_sol = sympy.dsolve(ode,ics=ics)
sympy.pprint(ode_sol,num_columns = 1000)

#仅含有一个独立变量的微分方程
####################################     常微分方程       #############################################
'''scipy.integrate.odeint(func, y0, t, args=(), Dfun=None, col_deriv=0, full_output=0, 
        ml=None, mu=None, rtol=None, atol=None, tcrit=None, h0=0.0, hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, 
        mxhnil=0, mxordn=12, mxords=5, printmessg=0, tfirst=False)
        args：元组，可选，传递给函数的额外参数'''
def func(y,t):
    y = y   #如果是二阶，就需要降阶，然后把所有一阶导放左边，表达式放下面
    #y1 ,y2 = y
    dy_dt = [1]  #需要总结成‘一阶导=表达式’这类形式，列表里就是表达式的集合体
    return dy_dt
t = np.linspace(0,10,11)
y = odeint(func,0,t)
print(y)

###################################     数学求解     ############################################
'''scipy.integrate.solve_ivp(fun, t_span, y0, method='RK45', 
    t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)
'''
def fun1(t, y):
    #y1 = y[0]
    #y2 = y[1]
    y1 ,y2 = y  #左边的导数
    dy_dt = [y2, t*t - y2+t]  #右边的表达式
    return dy_dt

# 初始条件
y0 = [0, 1]
yy = solve_ivp(fun1, (0, 500), y0, method='Radau', t_eval=np.arange(1, 500, 1))
t = yy.t
data = yy.y
plt.plot(t, data[0, :])
plt.plot(t, data[1, :])
plt.xlabel("时间s")
plt.legend(["y1",'y2'])
plt.show()