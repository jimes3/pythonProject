import pylab
import numpy as np

if __name__ == "__main__":
  x = np.arange(1, 31, 1)
  y = np.array([20, 23, 26, 29, 32, 35, 38, 45, 53, 62, 73, 86, 101, 118, 138, 161, 188, 220, 257, 300, 350, 409, 478, 558, 651, 760, 887, 1035, 1208, 1410])
  z1 = np.polyfit(x, y, 3)              # 曲线拟合，返回值为多项式的各项系数
  p1 = np.poly1d(z1)                    # 返回值为多项式的表达式，也就是函数式子
  print('多项式:\n',p1)
  y_pred = p1(x)                        # 根据函数的多项式表达式，求解 y
  print('求解值:',np.polyval(p1, 29))             #根据多项式求解特定 x 对应的 y 值
  #print(np.polyval(z1, 29))             #根据多项式求解特定 x 对应的 y 值

  plot1 = pylab.plot(x, y, '*', label='original values')
  plot2 = pylab.plot(x, y_pred, 'r', label='fit values')
  pylab.title('')
  pylab.xlabel('')
  pylab.ylabel('')
  pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
  pylab.show()
  #pylab.savefig('p1.png', dpi=200, bbox_inches='tight')