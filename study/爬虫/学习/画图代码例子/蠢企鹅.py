import turtle as t

def infoPrt():
    print('coordinate: ' + str(t.pos()))
    print('angle: ' + str(t.heading()))

t.pensize(3)
t.hideturtle()
t.colormode(255)
t.color("gray")
t.setup(700, 650)
t.speed(10)
t.st()
t.pu()
t.goto(-210,86)
t.pd()
infoPrt()

# 头
print('头')
t.seth(85)
t.circle(-100,25)
infoPrt()

t.seth(58)
t.circle(-170,120)
infoPrt()

t.seth(300)
t.circle(-100,25)
infoPrt()

#手
print('手')
t.seth(10)
t.circle(-120,20)
infoPrt()

t.seth(340)
t.circle(-30,70)
infoPrt()

t.seth(265)
t.circle(-30,40)
infoPrt()

t.seth(215)
t.circle(-100,30)
infoPrt()

#身子
print('身子')
t.seth(290)
t.circle(-150,40)
infoPrt()

#底
print('底')
t.seth(240)
t.circle(-80,40)
infoPrt()

t.seth(198)
t.circle(-300,11)
infoPrt()

t.seth(187)
t.circle(-500,14)
infoPrt()

t.seth(171)
t.circle(-300,11)
infoPrt()

t.seth(159)
t.circle(-80,40)
infoPrt()

#身子
print('身子')
t.seth(120)
t.circle(-150,40)
infoPrt()

#手
print('手')
t.seth(180)
t.circle(-120,20)
infoPrt()

t.seth(155)
t.circle(-30,70)
infoPrt()

t.seth(82)
t.circle(-28,45)
infoPrt()

t.seth(35)
t.circle(-110,34)
infoPrt()

#脸
print('脸')
t.seth(340)
t.circle(450,41)
infoPrt()

t.penup()
t.goto(50,75)
t.pendown()


#眼睛
print('眼睛')
t.seth(80)
t.circle(80,40)
infoPrt()

t.seth(125)
t.circle(40,140)
infoPrt()

t.seth(170)
t.circle(100,20)
infoPrt()

t.seth(100)
t.circle(40,140)
infoPrt()

t.seth(245)
t.circle(80,40)
infoPrt()

#眼珠
print('眼珠')
t.penup()
t.goto(-100,100)
t.color('black')
t.pendown()
t.begin_fill()
t.fillcolor('black')
t.circle(6)
t.end_fill()
infoPrt()

t.penup()
t.goto(-5,100)
t.color('black')
t.pendown()
t.begin_fill()
t.fillcolor('black')
t.circle(6)
t.end_fill()
infoPrt()

#星星棒
print('星星棒')
t.penup()
t.goto(148,-10)
t.color('yellow')
t.pendown()
t.begin_fill()
t.fillcolor('yellow')
t.seth(10)
t.fd(25)
t.seth(100)
t.fd(110)
t.right(125)
t.fd(12)
t.left(144)
t.fd(23)
t.right(72)
t.fd(23)
t.left(144)
t.fd(23)
t.right(72)
t.fd(23)
t.left(144)
t.fd(23)
t.right(72)
t.fd(23)
t.left(144)
t.fd(23)
t.right(72)
t.fd(23)
t.left(144)
t.fd(12)
t.right(130)
t.fd(110)
t.end_fill()
infoPrt()

t.done()
