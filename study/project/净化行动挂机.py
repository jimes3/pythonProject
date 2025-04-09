import pyautogui
import pydirectinput
import time
pyautogui.PAUSE = 0
#适用于净化行动，关闭自动匹配队友，开启单人匹配，在开始匹配页面运行程序。
#########   以管理员身份运行！！！！    ###########
#局内动作循环  666
def repeat_keys(keys):
    try:
        a = time.time()
        bb = time.time()
        while a-bb+5000>0:
            for key in keys:
                pyautogui.keyDown(key)
                time.sleep(1)
                pyautogui.keyUp(key)
                time.sleep(9)
                bb = time.time()
        pyautogui.press('esc')
    except KeyboardInterrupt:
        print("Stopped by user")
#检测图片
def identify_picture(a):
    left, top, width, height = pyautogui.locateOnScreen(a,confidence=0.9,grayscale=True)   # 寻找图片
    center = pyautogui.center((left, top, width, height))    # 寻找图片的中心
    print('开始',center)
    pydirectinput.moveTo(center[0],center[1])
    pydirectinput.click()
    time.sleep(10)

# 4：3比例 + 无边框窗口
while True:
    try:
        identify_picture('./source/sta1.png') #检测开始匹配
    except TypeError:
        try:
            identify_picture('./source/bac.png') #检测返回大厅
        except TypeError:
            try:
                identify_picture('./source/img.png') #检测准备完毕
                repeat_keys(['w', 'a', 's', 'd']) #局内动作，防止强退
            except TypeError:
                try:
                    identify_picture('./source/img_1.png') #检测返回大厅（出错情况）
                except TypeError:
                    try:
                        identify_picture('./source/img_2.png') #检测确定（出错情况）
                    except TypeError:
                        try:
                            identify_picture('./source/img_3.png') #检测点击空白处关闭（出错情况）
                        except TypeError:
                            print('无')
                            time.sleep(2)
