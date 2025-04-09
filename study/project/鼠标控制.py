import pyautogui
import time
import keyboard
import cv2
import numpy as np
pyautogui.PAUSE = 0
def key_press(event):
    if event.name == ';':
        prev_gray = None
        while True:
            screenshot = pyautogui.screenshot(region=(1275, 780, 10, 40))
            # 将截屏转换为灰度
            gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
            # 第一次循环不计算MSE
            if prev_gray is None:
                prev_gray = gray
                continue
            # 计算均方误差
            mse = np.mean((gray - prev_gray) ** 2)
            if mse <= 4:
                continue
            else:
                pyautogui.click()  # 点击
                pyautogui.mouseUp()  # 释放
                break
            prev_gray = gray
keyboard.on_press(key_press)
# 设置"Enter"键为按键
keyboard.add_hotkey(';',lambda:None, suppress=True)
# 进入事件循环
keyboard.wait()

'''
#获取鼠标的实时位置
try:
    while True:
        x,y = pyautogui.position()
        rgb = pyautogui.screenshot().getpixel((x, y))
        posi = 'x:' + str(x).rjust(4) + ' y:' + str(y).rjust(4) + '  RGB:' + str(rgb)
        print('\r',posi,end='')
        time.sleep(0.5)
except KeyboardInterrupt:
    print('已退出！')
'''


'''
duration 的作用是设置移动时间，所有的gui函数都有这个参数，而且都是可选参数
移动到指定位置
pyautogui.moveTo(100,300,duration=1)
按方向移动
pyautogui.moveRel(100,500,duration=4)   # 第一个参数是左右移动像素值，第二个是上下
获取鼠标位置
print(pyautogui.position())   # 得到当前鼠标位置；输出：Point(x=200, y=800)

# 点击鼠标,下面的点击方式都不会释放，需要自己释放鼠标
pyautogui.click(10,10)   # 鼠标点击指定位置，默认左键
pyautogui.click(10,10,button='left')  # 单击左键
pyautogui.click(1000,300,button='right')  # 单击右键
pyautogui.click(1000,300,button='middle')  # 单击中间
双击鼠标
pyautogui.doubleClick(10,10)  # 指定位置，双击左键
pyautogui.rightClick(10,10)   # 指定位置，双击右键
pyautogui.middleClick(10,10)  # 指定位置，双击中键

点击和释放
pyautogui.mouseDown()   # 鼠标按下
pyautogui.mouseUp()    # 鼠标释放
拖动到指定位置
pyautogui.dragTo(100,300,duration=1)
按方向拖动
pyautogui.dragRel(100,500,duration=4)   # 第一个参数是左右移动像素值，第二个是上下
鼠标滚动
pyautogui.scroll(300)  # 向上滚动300个单位

获取截屏
im = pyautogui.screenshot()：返回屏幕的截图，是一个Pillow的image对象
im.getpixel((500, 500))：返回im对象上，（500，500）这一点像素的颜色，是一个RGB元组
pyautogui.pixelMatchesColor(500,500,(12,120,400)) ：是一个对比函数，对比的是屏幕上（500，500）这一点像素的颜色，与所给的元素是否相同
识别图像
# 图像识别（一个）
btm = pyautogui.locateOnScreen('zan.png')
print(btm)  # Box(left=1280, top=344, width=22, height=22)
# 图像识别（多个）
btm = pyautogui.locateAllOnScreen('zan.png')
print(list(btm))  # [Box(left=1280, top=344, width=22, height=22), Box(left=25, top=594, width=22, height=22)]
pyautogui.center((left, top, width, height)) 返回指定位置的中心点；这样，我们就可以再配合鼠标操作点击找到图片的中心

键盘输入
pyautogui.keyDown() ： 模拟按键按下；
pyautogui.keyUp() ： 模拟按键释放；
pyautogui.press() ： # 就是调用keyDown() & keyUp(),模拟一次按键；
pyautogui.typewrite('this',0.5) ： 第一参数是输入内容，第二个参数是每个字符间的间隔时间；
pyautogui.typewrite(['T','h','i','s'])：typewrite 还可以传入单字母的列表
使用 pyautogui.hotkey()，这个函数可以接受多个参数，按传入顺序按下，再按照相反顺序释放
pyautogui.hotkey('ctrl','c')

'''