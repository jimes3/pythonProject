import turtle 
turtle.setup(800, 800, 200, 200) 
turtle.pencolor("yellow") 
turtle.pensize(4) 
turtle.penup()
 #turtle库准备
    
turtle.fd(-150) 
turtle.pendown()
 
'''
花的绘制, （旋转角度）100*18（旋转次数），
刚好为360度的5倍，因而能够闭合。大家
也可以尝试改变这两个值，得到不同效果。
'''
for i in range(18): 
     turtle.fd(300) 
     turtle.left(100)
 
#茎秆部分，移动画笔到合适位置
turtle.fd(150) 
turtle.right(90) 
turtle.pensize(8) 
turtle.pencolor("green") 
turtle.fd(400) 
turtle.penup() 
turtle.pensize(6) 
turtle.pendown() 

#叶子的绘制
turtle.fd(-250) 
turtle.seth(45) 
turtle.circle(-130,60) 
turtle.seth(-135) 
turtle.circle(-130,60) 
turtle.seth(135) 
turtle.circle(130,60) 
turtle.seth(-45) 
turtle.circle(130,60) 
turtle.done()