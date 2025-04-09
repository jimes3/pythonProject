import turtle
colors = ["red","orange","yellow","green","blue","indigo","purple",'black']#颜色
for i in range(len(colors)):#循环画8个
    c = colors[i]
    turtle.color(c)#显示颜色
    turtle.begin_fill()
    turtle.circle(100)
    turtle.rt(360/len(colors))#分割八次
    turtle.end_fill()
turtle.done()
